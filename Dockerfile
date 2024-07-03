FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y sox libsox-fmt-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# ENV PYTHONPATH="/app/src/SpeechText/IndicTransTokenizer:${PYTHONPATH}"

EXPOSE 80

CMD ["python", "main.py"]

