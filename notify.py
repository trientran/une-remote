#!/usr/bin/env python3
"""Send a run-status email with metrics + error tail inline, and files attached.

Usage:
    python notify.py <STATUS> <LOG_PATH> [RESULTS_JSON_PATH]

If RESULTS_JSON_PATH is omitted it falls back to $CARMACK_RESULTS or
results_baseline.json.

Credentials come from the environment (never hardcode secrets in a repo file):
    CARMACK_EMAIL_FROM           gmail address to send from       (required)
    CARMACK_EMAIL_APP_PASSWORD   gmail app password               (required)
    CARMACK_EMAIL_TO             recipient (default: same as FROM)
Set them once, e.g. in ~/.bashrc:
    export CARMACK_EMAIL_FROM='you@gmail.com'
    export CARMACK_EMAIL_APP_PASSWORD='xxxx xxxx xxxx xxxx'
    export CARMACK_EMAIL_TO='ttran72@myune.edu.au'
"""
import os, sys, json, socket, smtplib, mimetypes
from datetime import datetime
from email.message import EmailMessage

STATUS   = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
LOG_PATH = sys.argv[2] if len(sys.argv) > 2 else ""
RESULTS  = (sys.argv[3] if len(sys.argv) > 3 else
            os.environ.get("CARMACK_RESULTS",
                           "/scratch/ttran72/checkpoints/results_baseline.json"))

FROM = os.environ.get("CARMACK_EMAIL_FROM")
PW   = os.environ.get("CARMACK_EMAIL_APP_PASSWORD")
TO   = os.environ.get("CARMACK_EMAIL_TO", FROM)

if not FROM or not PW:
    sys.exit("notify.py: set CARMACK_EMAIL_FROM and CARMACK_EMAIL_APP_PASSWORD "
             "in the environment (see the module docstring). Not sending.")

host = socket.gethostname()
now  = datetime.now().strftime("%Y-%m-%d %H:%M")

lines = [f"Status : {STATUS}", f"Host   : {host}", f"Time   : {now}",
         f"Results: {os.path.basename(RESULTS)}", ""]

if os.path.isfile(RESULTS):
    try:
        data = json.load(open(RESULTS))
        lines.append("=== results (summary) ===")
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            for name, r in data.items():
                t = r.get("test", {}) if isinstance(r.get("test"), dict) else {}
                # notebook 01/02 store test accs either as {"accs":{k:..}} or flat "topK"
                accs = t.get("accs", {}) if isinstance(t.get("accs"), dict) else \
                       {k.replace("top", ""): v for k, v in t.items() if k.startswith("top")}
                acc_str = "  ".join(f"top{k}={float(accs[k]):.4f}"
                                    for k in sorted(accs, key=lambda x: int(x))) if accs else ""
                extra = []
                if r.get("val_top1_best") is not None: extra.append(f"val_top1={r['val_top1_best']:.4f}")
                elif r.get("best_acc") is not None:    extra.append(f"val_top1={r['best_acc']:.4f}")
                if t.get("macro_f1") is not None:      extra.append(f"macroF1={t['macro_f1']:.4f}")
                if t.get("weighted_f1") is not None:   extra.append(f"wF1={t['weighted_f1']:.4f}")
                if r.get("params") is not None:        extra.append(f"params={r['params']:,}")
                lines.append(f"- {name}: {acc_str}  {'  '.join(extra)}".rstrip())
        else:
            dump = json.dumps(data, indent=2)
            lines.append(dump[:6000] + ("\n...(truncated; see attachment)" if len(dump) > 6000 else ""))
    except Exception as e:
        lines.append(f"(could not parse results json: {e})")
else:
    lines.append("results file: NOT FOUND (run did not reach the final write)")

if STATUS.startswith("FAIL") and LOG_PATH and os.path.isfile(LOG_PATH):
    try:
        tail = open(LOG_PATH, errors="replace").read().splitlines()[-80:]
        lines += ["", "=== log tail (last 80 lines) ===", *tail]
    except Exception as e:
        lines.append(f"(could not read log: {e})")

msg = EmailMessage()
msg["Subject"] = f"[Carmack] training {STATUS} - {now}"
msg["From"] = FROM
msg["To"]   = TO
msg.set_content("\n".join(lines))

for path in (RESULTS, LOG_PATH):
    if path and os.path.isfile(path):
        ctype, _ = mimetypes.guess_type(path)
        maintype, subtype = (ctype.split("/", 1) if ctype else ("application", "octet-stream"))
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                               filename=os.path.basename(path))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
    s.login(FROM, PW)
    s.send_message(msg)
print(f"notification sent: {STATUS}")
