"""
Email notification service for NASDAQ ML pipeline alerts.

Sends email alerts when the daily automation pipeline fails,
including error details and the log file as an attachment.
"""

import os
import ssl
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _get_config() -> dict:
    """Load email configuration from environment variables."""
    return {
        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'sender_email': os.getenv('ALERT_EMAIL_FROM', ''),
        'sender_password': os.getenv('ALERT_EMAIL_PASSWORD', ''),
        'recipient_email': os.getenv('ALERT_EMAIL_TO', ''),
    }


def is_email_configured() -> bool:
    """Check if email alerting is configured."""
    config = _get_config()
    return bool(config['sender_email'] and config['sender_password'] and config['recipient_email'])


def send_failure_alert(
    error_message: str,
    step_name: str = "Unknown",
    log_file_path: Optional[str] = None,
) -> bool:
    """
    Send an email alert when the NASDAQ ML pipeline fails.

    Args:
        error_message: Description of the failure.
        step_name: The pipeline step that failed.
        log_file_path: Optional path to the log file to attach.

    Returns:
        True if the email was sent successfully, False otherwise.
    """
    if not is_email_configured():
        logger.warning("[EMAIL] Email alerting not configured — skipping alert. "
                       "Set ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_TO in .env")
        return False

    config = _get_config()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    subject = f"[NASDAQ ML ALERT] Pipeline Failed — {step_name} ({datetime.now().strftime('%Y-%m-%d')})"

    body = f"""NASDAQ ML Daily Automation — Failure Alert
==========================================

Time:           {now}
Failed Step:    {step_name}

Error Details:
--------------
{error_message}

-----
Log file: {log_file_path or 'N/A'}
Machine:  {os.environ.get('COMPUTERNAME', 'unknown')}

This is an automated alert from the NASDAQ 100 ML pipeline.
"""

    return _send_email(
        config=config,
        subject=subject,
        body=body,
        attachment_path=log_file_path,
    )


def send_success_summary(
    summary: str,
    log_file_path: Optional[str] = None,
) -> bool:
    """Send a success summary email (optional, for daily digest)."""
    if not is_email_configured():
        return False

    config = _get_config()
    subject = f"[NASDAQ ML] Daily Run Completed — {datetime.now().strftime('%Y-%m-%d')}"

    return _send_email(
        config=config,
        subject=subject,
        body=summary,
        attachment_path=log_file_path,
    )


def _send_email(
    config: dict,
    subject: str,
    body: str,
    attachment_path: Optional[str] = None,
) -> bool:
    """Send an email via SMTP with optional file attachment."""
    try:
        msg = MIMEMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = config['recipient_email']
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        if attachment_path:
            log_path = Path(attachment_path)
            if log_path.exists() and log_path.stat().st_size < 5 * 1024 * 1024:  # <5 MB
                with open(log_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{log_path.name}"',
                )
                msg.attach(part)

        context = ssl.create_default_context()
        with smtplib.SMTP(config['smtp_server'], config['smtp_port'], timeout=30) as server:
            server.starttls(context=context)
            server.login(config['sender_email'], config['sender_password'])
            server.sendmail(config['sender_email'], config['recipient_email'], msg.as_string())

        logger.info(f"[EMAIL] Alert email sent to {config['recipient_email']}")
        return True

    except Exception as e:
        logger.error(f"[EMAIL] Failed to send alert email: {e}")
        return False
