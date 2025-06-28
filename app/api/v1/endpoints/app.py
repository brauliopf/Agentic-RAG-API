from fastapi import APIRouter
from pydantic import BaseModel
from ....core.redis_client import redis

router = APIRouter()

class RedisSetRequest(BaseModel):
    key: str
    value: str

@router.post("/redis/set")
async def set_redis(request: RedisSetRequest):
    redis.set(request.key, request.value)
    return {"status": "success", "message": "Value set in Redis."}

 