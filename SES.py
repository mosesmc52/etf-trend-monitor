import boto3
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class AmazonSES(object):

    def __init__(self, region, access_key, secret_key, from_address, charset="UTF-8"):
        self.region = self._required("AWS_SES_REGION_NAME", region)
        self.access_key = self._required("AWS_SES_ACCESS_KEY_ID", access_key)
        self.secret_key = self._required("AWS_SES_SECRET_ACCESS_KEY", secret_key)
        self.from_address = self._required("FROM_ADDRESS", from_address)
        self.client = boto3.client(
            "ses",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        self.CHARSET = charset

    @staticmethod
    def _required(name, value):
        if value is None:
            raise ValueError(f"Missing required environment variable: {name}")
        text = str(value).strip()
        if not text:
            raise ValueError(f"Missing required environment variable: {name}")
        return text

    def send_text_email(self, to_address, subject, content):

        response = self.client.send_email(
            Destination={
                "ToAddresses": [to_address],
            },
            Message={
                "Body": {
                    "Text": {
                        "Charset": self.CHARSET,
                        "Data": content,
                    }
                },
                "Subject": {
                    "Charset": self.CHARSET,
                    "Data": subject,
                },
            },
            Source=self.from_address,
        )

    def send_html_email(self, to_address, subject, content):
        response = self.client.send_email(
            Destination={
                "ToAddresses": [
                    to_address,
                ],
            },
            Message={
                "Body": {
                    "Html": {
                        "Charset": self.CHARSET,
                        "Data": content,
                    }
                },
                "Subject": {
                    "Charset": self.CHARSET,
                    "Data": subject,
                },
            },
            Source=self.from_address,
        )

    def send_html_email_with_attachment(
        self,
        to_address,
        subject,
        html_content,
        text_content,
        attachment_name,
        attachment_bytes,
    ):
        message = MIMEMultipart("mixed")
        message["Subject"] = subject
        message["From"] = self.from_address
        message["To"] = to_address

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(text_content or "", "plain", self.CHARSET))
        alt.attach(MIMEText(html_content or "", "html", self.CHARSET))

        body_part = MIMEMultipart("related")
        body_part.attach(alt)
        message.attach(body_part)

        attachment_part = MIMEApplication(attachment_bytes)
        attachment_part.add_header(
            "Content-Disposition",
            "attachment",
            filename=attachment_name,
        )
        message.attach(attachment_part)

        return self.client.send_raw_email(
            Source=self.from_address,
            Destinations=[to_address],
            RawMessage={"Data": message.as_string()},
        )
