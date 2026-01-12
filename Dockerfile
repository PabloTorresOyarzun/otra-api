FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY main.py .
COPY quality.py .
COPY patrones.py .
COPY clasificacion.py .
COPY token_manager.py .
COPY database.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]