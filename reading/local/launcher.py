"""rAidiology local launcher: serves site/ and a small status page on this computer only.
Usage: python reading/local/launcher.py [--port N] [--no-browser] [--exit-after SECONDS]
Standard library only (Python 3.9+). It never sends anything anywhere."""
import argparse
import http.server
import json
import os
import platform
import re
import shutil
import sys
import threading
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SITE = os.path.join(ROOT, "site")
UI = os.path.join(HERE, "ui")
WEIGHTS = (".safetensors", ".onnx", ".pt")
PORT = 8765
NOWHERE = os.path.join(SITE, "\0nothing")  # a path that never exists, so the server answers 404


def version():
    try:
        with open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.search(r"\d+\.\d+(?:\.\d+)?", line)
                if m and line.lstrip().startswith(("#", "v", "V", "-", "*")):
                    return m.group(0)
    except OSError:
        pass
    return None


def installed(key):
    d = os.path.join(ROOT, "models", key)
    try:
        return any(n.lower().endswith(WEIGHTS) for n in os.listdir(d))
    except (OSError, ValueError):
        return False


def status():
    try:
        with open(os.path.join(HERE, "models.json"), encoding="utf-8") as f:
            models = json.load(f)
    except (OSError, ValueError):
        models = []
    for m in models:
        m["installed"] = installed(m.get("key", ""))
    try:
        free = shutil.disk_usage(ROOT).free
    except OSError:
        free = None
    return {"version": version(), "python": platform.python_version(),
            "os": platform.platform(), "gpu": bool(shutil.which("nvidia-smi")),
            "disk_free_bytes": free, "models": models}


def safe_path(base, rel):
    """Map a URL path onto base, or None if it would leave base."""
    rel = urllib.parse.unquote(rel.split("?", 1)[0].split("#", 1)[0])
    if "\x00" in rel or "\\" in rel or ":" in rel:
        return None
    parts = rel.split("/")
    if ".." in parts:
        return None
    p = os.path.realpath(os.path.join(base, *[x for x in parts if x]))
    b = os.path.realpath(base)
    if p != b and not p.startswith(b + os.sep):
        return None
    return p


class Handler(http.server.SimpleHTTPRequestHandler):
    server_version = "rAidiologyLocal"

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def translate_path(self, path):
        raw = path.split("?", 1)[0]
        if raw.startswith("//"):
            return NOWHERE
        if raw == "/launcher" or raw.startswith("/launcher/"):
            base, rel = UI, raw[len("/launcher"):]
        else:
            base, rel = SITE, raw
        return safe_path(base, rel) or NOWHERE

    def list_directory(self, path):
        return self.send_error(404)

    def do_GET(self):
        raw = self.path.split("?", 1)[0]
        if raw == "/launcher":
            self.send_response(301)
            self.send_header("Location", "/launcher/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if raw == "/launcher/api/status":
            return self._json(200, status())
        if raw.startswith("/launcher/api/"):
            return self.send_error(404)
        dec = urllib.parse.unquote(raw)
        if ".." in dec.split("/") or "\\" in dec or raw.startswith("//"):
            return self.send_error(403)
        super().do_GET()

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/launcher/api/stop":
            return self.send_error(404)
        port = self.server.server_address[1]
        if self.headers.get("Origin") not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            return self._json(403, {"error": "refused"})
        self._json(200, {"stopped": True})
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description="Run rAidiology on this computer.")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--exit-after", type=float, default=None, metavar="SECONDS")
    a = ap.parse_args()
    ports = [a.port] if a.port else range(PORT, 8800)
    srv = None
    for p in ports:
        try:
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    if srv is None:
        print("Could not find a free port between 8765 and 8799. Close other programs and try again.")
        return 1
    port = srv.server_address[1]
    url = f"http://127.0.0.1:{port}/launcher/"
    print(f"rAidiology is running at {url}  (press Ctrl+C or the Stop button to quit)", flush=True)
    if a.exit_after:
        t = threading.Timer(a.exit_after, srv.shutdown)
        t.daemon = True
        t.start()
    if not a.no_browser and os.environ.get("RAIDIOLOGY_NO_BROWSER") != "1":
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
