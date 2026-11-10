"""Answer access requests to the paid models' repositories on Hugging Face, so nobody reviews them by hand.

The paid models (Schedule A of COMMERCIAL-LICENSE.md) go to Hugging Face with a manual gate (publish.py): a request
waits until it is answered. This answers each one by the email address Hugging Face shares with the request:

  approved   an address at a university or school (university_domains.txt, or an academic suffix such as .edu or
             .ac.uk), at a government or public health body (.gov, .gov.uk, .nhs.uk, .gc.ca, europa.eu ...), or at a
             domain in approved_domains.txt (a charity or public hospital that wrote in and was checked)
  declined   everything else, with DECLINE_REASON, which Hugging Face shows the person: a company gets the commercial
             licence, starting with a free 60-day evaluation copy; a charity or a person writes to contact@lunarwerx.com

What a person types into the form never approves anyone: anybody can type 'university'. A declined request whose
domain is later added to approved_domains.txt is approved on the next pass; an approval is never taken back here.

The repositories it answers for are the manual-gated ones under --org whose name is rAidiology-<paid card>, the name
publish.py gives them. Without --act it is a dry run: it prints each decision and changes nothing. --act answers, and
appends one line per answer (time, repository, user name, email domain, answer) to the log beside the timer.
--install-timer MINUTES registers a hidden Windows scheduled task that runs --act that often; publish.py calls it after
it pushes a paid model, so the answering starts with the first repository and needs no one to remember it.

Login: huggingface_hub's own (`hf auth login`); no token is read, printed or logged here."""
import argparse
import datetime
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNIVERSITIES = os.path.join(HERE, "university_domains.txt")
APPROVED = os.path.join(HERE, "approved_domains.txt")
ORG = "LunarWerx"
TASK = "rAidiology Hugging Face access review"
LOG = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "rAidiology", "access-review.log")

# Academic and public-body suffixes. Each is matched at the end of the domain after a dot (or as the whole domain), so
# fakeedu.com, edu.example.com and evilgov.uk never match.
PUBLIC_SUFFIX = re.compile(
    r"(?:^|\.)(?:"
    r"edu|gov|mil|int"                                            # US universities and government, and treaty bodies
    r"|(?:edu|ac|sch|gov|gob|gouv|govt|go|gv|mil|nhs)\.[a-z]{2}"   # edu.au, ac.uk, sch.uk, gov.uk, gob.mx, go.jp ...
    r"|nhs\.net|gc\.ca|canada\.ca|europa\.eu|k12\.[a-z]{2}\.us"
    r")$")

DECLINE_REASON = (
    "These rAidiology models are free on request for universities, schools, public research and health bodies and "
    "governments: ask from your institution's email address. A company needs the commercial licence, which starts "
    "with a free 60-day evaluation copy: email contact@lunarwerx.com from your work address. A charity or a person: "
    "email contact@lunarwerx.com.")


def read_domains(path):
    """One domain per line; '#' starts a comment. A missing file is an empty list."""
    if not os.path.isfile(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.split("#", 1)[0].strip().lower().lstrip(".") for line in f} - {""}


def domain_of(email):
    if not email or email.count("@") != 1:
        return None
    local, domain = email.strip().lower().rsplit("@", 1)
    domain = domain.rstrip(".")
    if not local or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+", domain):
        return None
    return domain


def under(domain, listed):
    """True when the domain is a listed domain or a subdomain of one (cs.ox.ac.uk under ox.ac.uk)."""
    parts = domain.split(".")
    return any(".".join(parts[i:]) in listed for i in range(len(parts) - 1))


def decide(email, universities, approved):
    """('accept', why) or ('reject', DECLINE_REASON) for one request's email address."""
    domain = domain_of(email)
    if domain is None:
        return "reject", DECLINE_REASON
    if under(domain, approved):
        return "accept", "approved_domains.txt"
    if PUBLIC_SUFFIX.search(domain):
        return "accept", "academic or public-body address"
    if under(domain, universities):
        return "accept", "university_domains.txt"
    return "reject", DECLINE_REASON


def paid_repos(api, org):
    sys.path.insert(0, HERE)
    from publish import paid_cards  # the licence's Schedule A is the one list of paid models
    names = {f"raidiology-{card}" for card in paid_cards()}
    repos = []
    for m in api.list_models(author=org):
        if m.id.split("/", 1)[1].lower() in names and api.model_info(m.id).gated == "manual":
            repos.append(m.id)
    return sorted(repos)


def log(line):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def review(org, act):
    from huggingface_hub import HfApi
    api = HfApi()
    universities, approved = read_domains(UNIVERSITIES), read_domains(APPROVED)
    repos = paid_repos(api, org)
    if not repos:
        print(f"no manual-gated paid repositories under {org}; nothing to answer")
        return 0
    answered = 0
    for repo in repos:
        waiting = [(r, "pending") for r in api.list_pending_access_requests(repo)]
        waiting += [(r, "rejected") for r in api.list_rejected_access_requests(repo)]
        for req, state in waiting:
            verdict, why = decide(req.email, universities, approved)
            if state == "rejected" and verdict == "reject":
                continue  # already declined, and still declined
            domain = domain_of(req.email) or "(no address)"
            print(f"{repo}  {req.username:<24} {domain:<32} {state:>8} -> {verdict}" + (f" ({why})" if verdict == "accept" else ""))
            if not act:
                continue
            if verdict == "accept":
                api.accept_access_request(repo, req.username)
            else:
                api.reject_access_request(repo, req.username, rejection_reason=DECLINE_REASON)
            answered += 1
            log(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\t{repo}\t{req.username}\t{domain}\t{verdict}")
    print(f"{'answered' if act else 'would answer'} {answered if act else 'the requests above'}")
    return answered


def install_timer(minutes):
    """A hidden Windows scheduled task that runs --act every MINUTES minutes (pythonw, so no window opens)."""
    if os.name != "nt":
        print(f"not Windows: run 'python {os.path.abspath(__file__)} --act' every {minutes} minutes another way")
        return False
    py = sys.executable
    pyw = os.path.join(os.path.dirname(py), "pythonw.exe")
    command = f'"{pyw if os.path.isfile(pyw) else py}" "{os.path.abspath(__file__)}" --act'
    subprocess.run(["schtasks", "/Create", "/F", "/SC", "MINUTE", "/MO", str(minutes), "/TN", TASK, "/TR", command],
                   check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print(f"scheduled task '{TASK}' answers requests every {minutes} minutes; its log is {LOG}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--org", default=ORG, help=f"the Hugging Face organisation (default {ORG})")
    ap.add_argument("--act", action="store_true", help="answer the requests; without it this is a dry run")
    ap.add_argument("--install-timer", type=int, metavar="MINUTES", help="register the hidden scheduled task and stop")
    a = ap.parse_args()
    if a.install_timer:
        if a.install_timer < 5:
            sys.exit("error: every 5 minutes at the most often")
        install_timer(a.install_timer)
        return
    try:
        review(a.org, a.act)
    except Exception as exc:  # a network or permission error; its kind and text, never anything that looks like a token
        text = re.sub(r"(hf_[A-Za-z0-9]{6,}|Bearer\s+\S+)", "***", str(exc))
        if a.act:
            log(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\terror\t{type(exc).__name__}: {text}")
        sys.exit(f"error: {type(exc).__name__}: {text}")


if __name__ == "__main__":
    main()
