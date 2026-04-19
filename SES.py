import boto3


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
