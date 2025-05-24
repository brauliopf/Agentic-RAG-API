from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

app = FastAPI()

@app.get("/", response_class=PlainTextResponse)
def read_root():
    return "hello world"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
