# Dockerfile

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# Nerfstudio + FastAPI stack
RUN pip install --no-cache-dir \
      nerfstudio \
      fastapi uvicorn[standard] pydantic

# App code
COPY nerf_core.py /opt/program/nerf_core.py
COPY inference_server.py /opt/program/inference_server.py

EXPOSE 8080

ENTRYPOINT ["python", "-u", "/opt/program/inference_server.py"]
