import imaplib
import email
from email.header import decode_header
from langchain.tools import tool
import os


def _decode_value(value):
    """Decodes MIME encoded words."""
    if value is None:
        return ""
    decoded, charset = decode_header(value)[0]
    if isinstance(decoded, bytes):
        return decoded.decode(charset or "utf-8", errors="ignore")
    return decoded


def _extract_body(msg):
    """Extracts plain text body from a multipart or single-part email."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                return part.get_payload(decode=True).decode("utf-8", errors="ignore")
    else:
        return msg.get_payload(decode=True).decode("utf-8", errors="ignore")
    return ""
    

@tool("fetch_latest_email", return_direct=True)
def fetch_latest_email(_=None):
    """
    Fetches the latest email from the inbox and returns its subject and body.
    """

    EMAIL_HOST = os.getenv("EMAIL_HOST")
    EMAIL_PORT = int(os.getenv("EMAIL_PORT", 993))
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

    try:
        mail = imaplib.IMAP4_SSL(EMAIL_HOST, EMAIL_PORT)
        mail.login(EMAIL_USER, EMAIL_PASSWORD)
        mail.select("inbox")

        result, data = mail.search(None, "ALL")
        mail_ids = data[0].split()

        if not mail_ids:
            return "No emails in inbox."

        latest_id = mail_ids[-1]

        result, msg_data = mail.fetch(latest_id, "(RFC822)")
        raw_msg = msg_data[0][1]
        msg = email.message_from_bytes(raw_msg)

        subject = _decode_value(msg["Subject"])
        body = _extract_body(msg)
        from_addr = _decode_value(msg.get("From"))
        return f"From: {from_addr}\n\nSubject: {subject}\n\nBody:\n{body}"

    except Exception as e:
        return f"Error fetching email: {e}"
    
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from langchain.tools import tool
import os


@tool("send_email", return_direct=True)
def send_email(to_addr: str, subject: str, body: str):
    """
    Sends an email using SMTP.
    
    Expected payload format:
    {
        "to": "founder@example.com",
        "subject": "Your pitch evaluation",
        "body": "Hi ... (LLM-generated body)"
    }
    """
    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASSWORD = os.getenv("EMAIL_PASSWORD")

    if not to_addr:
        return "Error: Missing 'to' field."
    if not subject:
        return "Error: Missing 'subject' field."
    if not body:
        return "Error: Missing 'body' field."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = formataddr(("Phoenix Capital Partners", SMTP_USER))
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_addr, msg.as_string())
        return f"Email sent successfully to {to_addr}."

    except Exception as e:
        return f"Error sending email: {e}"
