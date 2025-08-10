FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .

RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config python3-dev \
    && pip install --no-cache-dir -r backend/requirements.txt \
    && apt-get remove -y build-essential pkg-config python3-dev \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY backend/app ./backend/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
