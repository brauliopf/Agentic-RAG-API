from fastapi import APIRouter
from pydantic import BaseModel
from ....core.redis_client import redis
import boto3
from botocore.config import Config

router = APIRouter()

class RedisSetRequest(BaseModel):
    key: str
    value: str

@router.post("/redis/set")
async def set_redis(request: RedisSetRequest):
    redis.set(request.key, request.value)
    return {"status": "success", "message": "Value set in Redis."}

class GetS3UploadUrlRequest(BaseModel):
    bucket_name: str
    object_name: str
    expiration: int = 3600
    region_name: str = 'us-east-1'

@router.post("/s3/uploadUrl")
async def get_s3_upload_url(request: GetS3UploadUrlRequest):
    presigned_url = create_presigned_url(request.bucket_name, request.object_name, request.expiration, request.region_name)
    return {"status": "success", "message": "File uploaded to S3.", "presigned_url": presigned_url}
 
def create_presigned_url(bucket_name, object_name, expiration=3600, region_name='us-east-1'):
    """
    Generates a pre-signed URL for uploading an object to S3.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_name (str): The desired object key (path and filename) in the bucket.
        expiration (int): The number of seconds the pre-signed URL will be valid for.
        region_name (str): The AWS region where the bucket is located.

    Returns:
        str: The pre-signed URL, or None if an error occurred.
    """
    try:
        s3_client = boto3.client('s3', region_name=region_name, config=Config(signature_version='s3v4'))
        response = s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=expiration,
            # Add other parameters like 'ContentType' if needed
        )
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
    return response