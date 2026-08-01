#!/usr/bin/env python3
"""Fangstproxy — sparar den EXAKTA payload en agentklient skickar.

Varfor: att jamfora en klients interna meddelandeobjekt racker inte. Det ar
serialiseringen som avgor prefix-cachen — nyckelordning, whitespace, verktygens
ordning, metadata som ramverket lagger pa sist. Bara det som gar over tradet raknas.

Anvandning:
    python3 capture_proxy.py                    # lyssnar 8899 -> 127.0.0.1:8888
    python3 capture_proxy.py 8899 8000 ./fangst # egen port, uppstrom och katalog

Peka sedan agentramverket pa http://<host>:8899/v1 i stallet for :8888.
Varje request sparas som fangst/NNNN-<endpoint>.json, i ordning.

!! RA FANGSTER KAN INNEHALLA KOD, TOKENS, FILVAGAR OCH HEMLIGHETER.
   fangst/ ar gitignorerad. Halla originalen utanfor versionskontroll och dela
   dem inte. Kor analysen med REDACT=1 sa innehallssnuttar aldrig skrivs ut:
       REDACT=1 python3 prompt_locality.py fangst/0006-chat.json fangst/0007-chat.json

Efterat:
    python3 prompt_locality.py fangst/0006-chat.json fangst/0007-chat.json

Proxyn ar avsiktligt dum: den andrar ingenting, buffrar hela kroppen och
vidarebefordrar. Anvand den for matning, inte i produktion.
"""
import json, os, sys, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
UPSTREAM = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "fangst"
os.umask(0o077)          # filer och kataloger skapas utan grupp/varldsrattigheter fran borjan
os.makedirs(OUTDIR, exist_ok=True)
os.chmod(OUTDIR, 0o700)  # aven om katalogen redan fanns
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
            # spara exakt som skickat, bara indenterat for lasbarhet
            open(path, "wb").write(json.dumps(json.loads(body), ensure_ascii=False,
                                              indent=1).encode("utf-8"))
        except Exception:
            open(path, "wb").write(body)
        os.chmod(path, 0o600)
        d = "?"
        try:
            o = json.loads(body)
            d = "%d meddelanden, %d verktyg, ~%d tecken" % (
                len(o.get("messages", [])), len(o.get("tools", [])), len(body))
        except Exception:
            d = "%d bytes" % len(body)
        print("[%04d] %-28s %s" % (_n[0], self.path, d), flush=True)
        self._pass(body)

    def do_GET(self):
        self._pass()


print("fangstproxy: 0.0.0.0:%d  ->  127.0.0.1:%d" % (LISTEN, UPSTREAM))
print("sparar till: %s/" % OUTDIR)
print("peka klienten pa http://<host>:%d/v1 och kor din uppgift.\n" % LISTEN)
ThreadingHTTPServer(("0.0.0.0", LISTEN), H).serve_forever()
