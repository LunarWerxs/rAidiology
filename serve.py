"""Local preview of the site with caching off, so edits show on the next reload.
Usage: python serve.py [port]"""
import functools
import http.server
import os
import sys


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
site = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")
handler = functools.partial(NoCache, directory=site)
print(f"http://127.0.0.1:{port}/")
http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
