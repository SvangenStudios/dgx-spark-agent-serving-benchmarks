#!/usr/bin/env python3
"""Capture proxy — records the EXACT payload an agent client sends.

Why: comparing a client's internal message objects is not enough. Serialization
is what determines the prefix cache — key order, whitespace, tool ordering,
metadata the framework appends last. Only what goes over the wire counts.

Usage:
    python3 capture_proxy.py                      # listens 8899 -> 127.0.0.1:8888
    python3 capture_proxy.py 8899 8000 ./captures # custom port, upstream and dir

Then point the agent framework at http://<host>:8899/v1 instead of :8888.
Each request is saved as captures/NNNN-<endpoint>.json, in order.

!! RAW CAPTURES MAY CONTAIN CODE, TOKENS, FILE PATHS AND SECRETS.
   captures/ is gitignored. Keep originals out of version control and do not
   share them. Run the analysis with REDACT=1 so content snippets are never
   printed:
       REDACT=1 python3 prompt_locality.py captures/0006-chat.json captures/0007-chat.json

Afterwards:
    python3 prompt_locality.py captures/0006-chat.json captures/0007-chat.json

The proxy is deliberately dumb: it changes nothing, buffers the whole body and
forwards it. Use it for measurement, not in production.
"""
import json, os, sys, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
UPSTREAM = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "captures"
os.umask(0o077)          # files and dirs are created without group/world permissions
os.makedirs(OUTDIR, exist_ok=True)
os.chmod(OUTDIR, 0o700)  # even if the directory already existed
_n = [0]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _pass(self, body=None):
        url = "http://127.0.0.1:%d%s" % (UPSTREAM, self.path)
        hdrs = {k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "connection")}
        req = urllib.request.Request(url, body, hdrs, method=self.command)
        try:
            r = urllib.request.urlopen(req, timeout=3600)
            data = r.read()
            self.send_response(r.status)
            for k, v in r.headers.items():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            msg = json.dumps({"error": str(e)}).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        _n[0] += 1
        tag = self.path.rstrip("/").split("/")[-1] or "req"
        path = os.path.join(OUTDIR, "%04d-%s.json" % (_n[0], tag))
        try:
            # save exactly as sent, only indented for readability
            open(path, "wb").write(json.dumps(json.loads(body), ensure_ascii=False,
                                              indent=1).encode("utf-8"))
        except Exception:
            open(path, "wb").write(body)
        os.chmod(path, 0o600)
        d = "?"
        try:
            o = json.loads(body)
            d = "%d messages, %d tools, ~%d chars" % (
                len(o.get("messages", [])), len(o.get("tools", [])), len(body))
        except Exception:
            d = "%d bytes" % len(body)
        print("[%04d] %-28s %s" % (_n[0], self.path, d), flush=True)
        self._pass(body)

    def do_GET(self):
        self._pass()


print("capture proxy: 0.0.0.0:%d  ->  127.0.0.1:%d" % (LISTEN, UPSTREAM))
print("saving to: %s/" % OUTDIR)
print("point the client at http://<host>:%d/v1 and run your task.\n" % LISTEN)
ThreadingHTTPServer(("0.0.0.0", LISTEN), H).serve_forever()
