# Security policy

rAidiology is a scan viewer that runs in your browser, with a local launcher and an MCP server that run on your own computer. A bug in any of them could expose a scan, a report or a person's data, so we want to hear about it.

rAidiology is not a medical device. It is for education and research, never for diagnosis or treatment decisions, and this policy does not change that.

## What it exposes

- **The scan viewer.** A website that opens the pictures from your own scan in the browser. Nothing is uploaded, and it works offline after the first visit.
- **The local launcher** (`reading/local/launcher.py`). It serves the site on `127.0.0.1`, so only your own computer can reach it.
- **The MCP server** (`mcp/`). It lets your own AI assistant run the research models on your own machine.

## Reporting a vulnerability

If you find a security problem or a privacy leak (for example, a way for another program or web page to reach the local server or the MCP server, or a file that seems to hold patient data), **please report it privately** rather than in a public issue:

- **Preferred:** GitHub's private
  **[Report a vulnerability](https://github.com/LunarWerxs/rAidiology/security/advisories/new)**
  (the repository's *Security, Advisories* tab), or
- open a normal issue **asking us to contact you** (don't include the exploit details there), or
- message the maintainers on [Discord](https://discord.gg/PsWpeNUzhk) and ask for a private channel.

You can also reach LunarWerx at contact@lunarwerx.com.

Please include:

- what triggers it: the steps, the request or the file (describe it, and do not attach any scan or patient data),
- the commit you ran and your operating system, and
- what you observed: what was reachable, leaked or changed.

## Supported versions

The current version on the `main` branch is the supported one. Please update, and confirm the problem still happens, before you report it.

Thank you for helping keep rAidiology users safe.
