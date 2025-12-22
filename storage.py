"""
DigitalOcean Spaces storage client (S3-compatible).
"""

import os
import boto3
from botocore.config import Config

# Load from environment
SPACES_KEY = os.getenv("DO_SPACES_KEY")
SPACES_SECRET = os.getenv("DO_SPACES_SECRET")
SPACES_BUCKET = os.getenv("DO_SPACES_BUCKET", "3dbucket")
SPACES_REGION = os.getenv("DO_SPACES_REGION", "nyc3")
SPACES_ENDPOINT = os.getenv("DO_SPACES_ENDPOINT", f"https://{SPACES_REGION}.digitaloceanspaces.com")

# Initialize client
_client = None


def get_client():
    """Get or create S3 client for DO Spaces."""
    global _client
    if _client is None:
        _client = boto3.client(
            's3',
            region_name=SPACES_REGION,
            endpoint_url=SPACES_ENDPOINT,
            aws_access_key_id=SPACES_KEY,
            aws_secret_access_key=SPACES_SECRET,
            config=Config(signature_version='s3v4')
        )
    return _client


def upload_file(file_bytes: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    """
    Upload file bytes to bucket.

    Args:
        file_bytes: Raw file bytes
        key: Object key (path in bucket)
        content_type: MIME type

    Returns:
        Public URL to the file
    """
    client = get_client()
    client.put_object(
        Bucket=SPACES_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
        ACL='public-read'  # Make publicly readable
    )
    return f"https://{SPACES_BUCKET}.{SPACES_REGION}.digitaloceanspaces.com/{key}"


def download_file(key: str) -> bytes:
    """Download file from bucket."""
    client = get_client()
    response = client.get_object(Bucket=SPACES_BUCKET, Key=key)
    return response['Body'].read()


def delete_file(key: str):
    """Delete file from bucket."""
    client = get_client()
    client.delete_object(Bucket=SPACES_BUCKET, Key=key)


def list_files(prefix: str = "") -> list:
    """List files in bucket with optional prefix."""
    client = get_client()
    response = client.list_objects_v2(Bucket=SPACES_BUCKET, Prefix=prefix)
    return [obj['Key'] for obj in response.get('Contents', [])]


def list_files_detailed(prefix: str = "", limit: int = 1000) -> list:
    """List files with metadata (size, last modified)."""
    client = get_client()

    files = []
    continuation_token = None

    while len(files) < limit:
        params = {
            'Bucket': SPACES_BUCKET,
            'Prefix': prefix,
            'MaxKeys': min(1000, limit - len(files))
        }
        if continuation_token:
            params['ContinuationToken'] = continuation_token

        response = client.list_objects_v2(**params)

        for obj in response.get('Contents', []):
            files.append({
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'].isoformat(),
                'url': f"https://{SPACES_BUCKET}.{SPACES_REGION}.digitaloceanspaces.com/{obj['Key']}"
            })

        if not response.get('IsTruncated'):
            break
        continuation_token = response.get('NextContinuationToken')

    return files


def get_bucket_info() -> dict:
    """Get info about the configured bucket."""
    return {
        'name': SPACES_BUCKET,
        'region': SPACES_REGION,
        'endpoint': SPACES_ENDPOINT,
        'url': f"https://{SPACES_BUCKET}.{SPACES_REGION}.digitaloceanspaces.com"
    }


def list_prefixes(prefix: str = "", delimiter: str = "/") -> list:
    """List 'folders' (common prefixes) in bucket."""
    client = get_client()
    response = client.list_objects_v2(
        Bucket=SPACES_BUCKET,
        Prefix=prefix,
        Delimiter=delimiter
    )

    prefixes = []
    for p in response.get('CommonPrefixes', []):
        folder = p['Prefix'].rstrip('/')
        if '/' in prefix:
            folder = folder.split('/')[-1]
        prefixes.append(folder)

    return prefixes
