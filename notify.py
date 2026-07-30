#!/usr/bin/env python3
"""Send a run-status email with metrics + error tail inline, and files attached.
Usage: python notify.py <STATUS> <LOG_PATH>
Credentials come from environment (see run.sh / carmack_notify.env):
  CARMACK_EMAIL_FROM, CARMACK_EMAIL_APP_PASSWORD, CARMACK_EMAIL_TO (optional)
"""
import os, sys, json, socket, smtplib, mimetypes
from datetime import datetime
from email.message import EmailMessage

STATUS   = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
LOG_PATH = sys.argv[2] if len(sys.argv) > 2 else ""
RESULTS  = "/scratch/ttran72/checkpoints/results_baseline.json"

FROM = "aunhi55@gmail.com"
PW   = "ybzb qhug rpsg mvik"
TO   = "ttran72@myune.edu.au"

host = socket.gethostname()
now  = datetime.now().strftime("%Y-%m-%d %H:%M")

# ---- build a concise, mobile-friendly body ----
lines = [f"Status : {STATUS}", f"Host   : {host}", f"Time   : {now}", ""]

# Inline the key metrics if results.json parsed cleanly
if os.path.isfile(RESULTS):
    try:
        data = json.load(open(RESULTS))
        lines.append("=== results (summary) ===")
        # try a tidy per-model view; fall back to pretty dump
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            for name, r in data.items():
                t = r.get("test", {}) if isinstance(r.get("test"), dict) else {}
                accs = t.get("accs", {}) if isinstance(t, dict) else {}
                acc_str = "  ".join(f"top{k}={accs[k]:.4f}" for k in sorted(accs, key=lambda x: int(x))) if accs else ""
                extra = []
                if "best_acc" in r:   extra.append(f"val_top1={r['best_acc']:.4f}")
                if t.get("macro_f1") is not None:    extra.append(f"macroF1={t['macro_f1']:.4f}")
                if t.get("weighted_f1") is not None: extra.append(f"wF1={t['weighted_f1']:.4f}")
                if "params" in r:     extra.append(f"params={r['params']:,}")
                lines.append(f"- {name}: {acc_str}  {'  '.join(extra)}".rstrip())
        else:
            dump = json.dumps(data, indent=2)
            lines.append(dump[:6000] + ("\n...(truncated; see attachment)" if len(dump) > 6000 else ""))
    except Exception as e:
        lines.append(f"(could not parse results.json: {e})")
else:
    lines.append("results file: NOT FOUND (run did not reach the final write)")

# On failure, inline the tail of the log so you can read the error on your phone
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

# ---- attach full files for completeness ----
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