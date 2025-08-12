FROM python:3.10-slim

WORKDIR /app

# Install cmake and other build dependencies
# Install system dependencies required for dlib and face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    python3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt

# streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.enableCORS=false"]
