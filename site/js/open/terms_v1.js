// The donation permission, version 1: the words that ship (docs/DONATED_SCANS.md, "The consent panel").
// One exported HTML string, used by the panel and by every saved copy. Never edit these words in
// place: a change is a new file (terms_v2.js) and a new version, for new donations only.
//
// Two placeholders, filled by fillPanel(): {{DATE}} (today's date) and {{POLICY_URL}} (the policy link).
// The hash in the consent record is over PANEL_HTML exactly as written here, before they are filled.

export const TERMS_VERSION = 1;

const BTN = 'font-size:19px;min-width:12em;padding:12px 20px;';

export const PANEL_HTML = `<section class="donate-panel" style="font-size:19px;line-height:1.5">
<h2 style="font-size:1.3em"><strong>Donate this scan to help train rAidiology? (optional)</strong></h2>
<p>Nothing has been sent. If you donate, your browser first removes everything it can find that could identify you: names, ID numbers, dates, addresses, the hospital and doctors, and device serial numbers. It does not send a set of pictures if it finds printed words in any of them, and never takes scans of the head or neck. It can miss things, such as a name printed into a picture, so please look through the pictures below before you send. It then sends only the pictures and a few technical details (body part, pixel size, slice positions, scanner make and model, your sex and your age in years), locked so that only rAidiology can open them. The list of exactly what will be sent is below. Written reports are never sent.</p>
<p><strong>What we do with it.</strong> We use donated scans only to train and test rAidiology's computer models that read medical scans. Those models are part of products we may charge for. You will not be paid. We never sell your scan, never make it public, never use it for advertising, and never try to find out who you are. No one else can see it. Connections Global LLC, a company with the same owner as rAidiology, stores it for us under contract, on Amazon Web Services, only as encrypted files it has no key to (the third box below asks you about this). Connections also carries the upload and sees your internet address, as any website does; it is not kept. If rAidiology ever changes hands, your scan goes only to a buyer who keeps these terms.</p>
<p><strong>What we keep.</strong> The scan as sent, the version of these terms, the month and year, and a code made from your donation number (not the number itself). Not your name, email or internet address.</p>
<p><strong>You can change your mind.</strong> Keep your donation number. With it you can see, save or delete your scan at any time on this site's donation page, or ask by writing to contact@lunarwerx.com. Deleting it withdraws your consent: we stop using and storing it, and nothing more is ever collected from you unless you donate again. A deleted scan is gone from our storage within 30 days and is never used in training again. A model that has already learnt from it cannot unlearn it. If you lose the number, open the same scan on that page: it makes a code from the pictures that lets us find and delete the donation.</p>
<p><strong>This permission.</strong> It lets your browser, running rAidiology's viewer, send this one scan to LunarWerx LLC (LunarWerx Studios), which makes rAidiology, for the uses above and no others. It ends once this scan is sent, and in any case one year from today; it never covers another scan. You have the right to a copy of it: after you donate you can save one, and get another at any time on this site's donation page with your donation number. Full details: <a href="{{POLICY_URL}}">Consumer Health Data Privacy Policy</a>.</p>
<p>Date: {{DATE}}</p>
<p><label><input type="checkbox" id="donate-eligible"> I am 18 or older, I live in the United States but not in Maryland or New York, and this is a scan of me.</label></p>
<p><label><input type="checkbox" id="donate-agree"> I agree to donate this scan on these terms. Ticking this box is my electronic signature to this permission; it is used for nothing else.</label></p>
<p><label><input type="checkbox" id="donate-storage"> I agree that Connections Global LLC, a company with the same owner as rAidiology, stores the locked files for rAidiology, on Amazon Web Services, only so that rAidiology can receive them, under the same terms. I can withdraw this by deleting my donation.</label></p>
<p><button type="button" id="donate-send" style="${BTN}"><strong>Donate this scan</strong></button> <button type="button" id="donate-no" style="${BTN}"><strong>No thanks</strong></button></p>
</section>`;

// SHA-256 (lower-case hex) of PANEL_HTML as written, before the placeholders are filled.
export async function panelSha256() {
  const bytes = new TextEncoder().encode(PANEL_HTML);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

// The panel with today's date and the policy link filled in.
export function fillPanel(dateText, policyUrl) {
  return PANEL_HTML.split('{{DATE}}').join(dateText).split('{{POLICY_URL}}').join(policyUrl);
}
