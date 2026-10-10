// Where the donation service lives: ONE constant, used by the donate panel and the donation page.
// The host is a walled corner of Connections on AWS (docs/DONATED_SCANS.md, "Where donations are
// stored"), not this site. While the host still holds the placeholder, nothing is shown or called.

export const DONATION_API = 'https://__DONATION_HOST__/api/donation';

// donate_crypto.js adds /api/donation/<route> to a base, so its base is the API without that tail.
export const DONATION_BASE = DONATION_API.replace(/\/api\/donation$/, '');

export const donationConfigured = (api = DONATION_API) => !api.includes('__DONATION_HOST__');

/** fetch for the donation host: no cookies, no auth, no referrer. */
export const donationFetch = (url, init = {}) =>
  globalThis.fetch(url, { ...init, credentials: 'omit', referrerPolicy: 'no-referrer' });
