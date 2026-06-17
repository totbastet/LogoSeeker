FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-gpu.txt ./

# Флаг для GPU (по умолчанию false)
ARG USE_GPU=false

RUN if [ "$USE_GPU" = "true" ] ; then \
      pip install --no-cache-dir -r requirements-gpu.txt ; \
    else \
      pip install --no-cache-dir -r requirements.txt ; \
    fi

COPY . .

RUN sed -i 's/weights_only=True/weights_only=False/g' /usr/local/lib/python3.11/site-packages/ultralytics/nn/tasks.py
RUN sed -i 's/weights_only: bool = True/weights_only: bool = False/g' /usr/local/lib/python3.11/site-packages/torch/serialization.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]