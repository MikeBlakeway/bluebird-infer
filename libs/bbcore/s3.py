"""S3/MinIO client wrapper for Bluebird inference pods.

Provides simple helpers for uploading/downloading objects, generating
presigned URLs, and building canonical project/take paths.
"""

from __future__ import annotations

import io
import typing as t

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import get_settings


class S3:
    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region_name: str = "us-east-1",
    ) -> None:
        # Path-style addressing helps with MinIO/local setups
        cfg = BotoConfig(s3={"addressing_style": "path"}, signature_version="s3v4")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
            config=cfg,
        )
        self.bucket = bucket

    # ---------- Bucket ----------
    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            code = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if code == 404:
                self._client.create_bucket(Bucket=self.bucket)
            elif code == 301:
                # Some providers require LocationConstraint; try default region
                self._client.create_bucket(
                    Bucket=self.bucket, CreateBucketConfiguration={"LocationConstraint": "us-east-1"}
                )
            else:
                raise

    # ---------- Keys & Paths ----------
    @staticmethod
    def build_key(project_id: str, take_id: str, relative: str) -> str:
        return f"projects/{project_id}/takes/{take_id}/{relative}".strip("/")

    # ---------- Upload ----------
    def upload_file(self, local_path: str, key: str, content_type: str | None = None) -> str:
        extra = {"ContentType": content_type} if content_type else None
        self._client.upload_file(local_path, self.bucket, key, ExtraArgs=extra)
        return f"s3://{self.bucket}/{key}"

    def upload_bytes(self, data: bytes, key: str, content_type: str | None = None) -> str:
        args = {"Bucket": self.bucket, "Key": key, "Body": data}
        if content_type:
            args["ContentType"] = content_type
        self._client.put_object(**args)
        return f"s3://{self.bucket}/{key}"

    # ---------- Download ----------
    def download_file_to_path(self, key: str, local_path: str) -> None:
        self._client.download_file(self.bucket, key, local_path)

    def download_bytes(self, key: str) -> bytes:
        obj = self._client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    # ---------- Presign ----------
    def presign(self, key: str, expires_in: int = 3600, method: str = "get_object") -> str:
        return self._client.generate_presigned_url(
            ClientMethod=method,
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )


def from_env() -> S3:
    s = get_settings()
    client = S3(
        endpoint_url=s.BB_S3_ENDPOINT,
        access_key=s.BB_S3_ACCESS_KEY,
        secret_key=s.BB_S3_SECRET,
        bucket=s.BB_BUCKET,
    )
    return client


__all__ = ["S3", "from_env"]
