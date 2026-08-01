import json, time, urllib.request, threading

U = "http://127.0.0.1:8888/v1/chat/completions"


def call(txt, mt=80, timeout=2400):
    p = {"model": "deepseek-v4-flash-0731",
         "messages": [{"role": "user", "content": txt}],
         "max_tokens": mt, "temperature": 0, "stream": False}
    r = urllib.request.Request(U, json.dumps(p).encode(), {"Content-Type": "application/json"})
    t = time.time()
    d = json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    e = time.time() - t
    return d["choices"][0]["message"].get("content") or "", d["usage"]["prompt_tokens"], e


FILL = ("Operations note: routine check of cooling and network links completed without remarks. "
        "Log rotation verified. Backup finished on schedule. No alarms registered during the shift. ")
DIST = "Reference code QX-00000 is invalid and must be ignored. "

N_START = "KEY-ALPHA-4471"
N_MID = "KEY-BETA-8823"
N_END = "KEY-GAMMA-1590"


def build(target_tokens):
    chunk = FILL * 8 + DIST
    per = int(len(chunk) / 3.6)          # ~3.6 chars per token for this text
    n = max(4, target_tokens // per)
    body = []
    for i in range(n):
        body.append(chunk)
        if i == int(n * 0.08):
            body.append("\nIMPORTANT: the first code is " + N_START + ".\n")
        if i == int(n * 0.50):
            body.append("\nIMPORTANT: the second code is " + N_MID + ".\n")
        if i == int(n * 0.92):
            body.append("\nIMPORTANT: the third code is " + N_END + ".\n")
    q = ("\n\nQuestion: list the three codes marked IMPORTANT, in order, "
         "separated by commas. Reply with only the codes.")
    return "".join(body) + q


print("depth    prompt_tok    hits      TTFT        prefill", flush=True)
for target in (32000, 128000, 512000, 900000):
    txt = build(target)
    try:
        c, pt, e = call(txt)
        hits = sum(1 for v in (N_START, N_MID, N_END) if v in c)
        print("%4dK   %10s    %d/3      %5.2f min   %6.0f tok/s"
              % (target // 1000, format(pt, ","), hits, e / 60, pt / e), flush=True)
        if hits < 3:
            print("        answer: " + c[:120].replace("\n", " "), flush=True)
    except Exception as ex:
        print("%4dK   FEL: %s %s" % (target // 1000, type(ex).__name__, str(ex)[:90]), flush=True)

print("\n=== CONCURRENCY: 1x long (500K) + 3x short ===", flush=True)
longtxt = build(500000)
lat = []
done = []


def short(i):
    t = time.time()
    call("Write one sentence about the sea.", mt=40)
    lat.append(time.time() - t)


def longreq():
    try:
        c, pt, e = call(longtxt)
        done.append("%s tok pa %.1f s" % (format(pt, ","), e))
    except Exception as ex:
        done.append("FEL " + str(ex)[:60])


tl = threading.Thread(target=longreq)
tl.start()
time.sleep(25)
ts = [threading.Thread(target=short, args=(i,)) for i in range(3)]
for t in ts:
    t.start()
for t in ts:
    t.join()
tl.join()
if lat:
    print("short requests during long: " + ", ".join("%.1fs" % x for x in lat)
          + "   max=%.1fs" % max(lat), flush=True)
print("long request: " + str(done), flush=True)
