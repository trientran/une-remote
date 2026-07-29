import smtplib, sys, socket
from email.message import EmailMessage

status = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
msg = EmailMessage()
msg["Subject"] = f"[Carmack] training {status}"
msg["From"] = "aunhi55@gmail.com"
msg["To"] = "ttran72@myune.edu.au"
msg.set_content(f"Run finished with status: {status}\nHost: {socket.gethostname()}")

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
    s.login("aunhi55@gmail.com", "ybzb qhug rpsg mvik")
    s.send_message(msg)
print("notification sent:", status)