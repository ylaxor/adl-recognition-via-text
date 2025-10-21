from collections.abc import Mapping
from email.message import EmailMessage
from pathlib import Path
from smtplib import SMTP

from toml import load as load_toml

config = load_toml(Path(__file__).parent.parent / "config" / "mailing.toml")


def send_email(
    subject: str,
    body: str,
    attachments: Mapping[str, str | bytes] | None = None,
) -> None:
    try:
        email_config = config["email"]
        sender = str(email_config.get("sender"))
        secret = str(email_config.get("secret"))
        recipient = str(email_config.get("recipient"))
        server = str(email_config.get("server"))
        port = int(email_config.get("port"))
    except Exception as e:
        print(f"Failed to load email configuration: {e}. Skipping email.")
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        msg.set_content(body)
        if attachments:
            for filename, content in attachments.items():
                if isinstance(content, bytes):
                    msg.add_attachment(
                        content,
                        filename=filename,
                        maintype="application",
                        subtype="octet-stream",
                    )
                else:
                    msg.add_attachment(
                        content,
                        filename=filename,
                    )
        with SMTP(server, port) as server:
            server.starttls()
            server.login(sender, secret)
            server.send_message(msg)
        print(f"Email sent to {recipient}.")
    except Exception as e:
        print(f"Failed to send email: {e}")
