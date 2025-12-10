# inference_server.py

from __future__ import annotations

import base64
import os

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from nerf_core import load_pipeline, render_camera_png_bytes

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")

app = FastAPI()
pipeline = None


class RenderRequest(BaseModel):
    camera_index: int = 0


@app.on_event("startup")
def startup_event():
    global pipeline
    print("[inference_server] Loading pipeline...")
    pipeline = load_pipeline(MODEL_DIR)
    print("[inference_server] Pipeline loaded.")


@app.post("/invocations")
def invoke(req: RenderRequest):
    png_bytes = render_camera_png_bytes(pipeline, camera_index=req.camera_index)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return {"image_base64": b64}


@app.get("/ping")
def ping():
    # Simple health check for SageMaker
    return {"status": "ok"}


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
