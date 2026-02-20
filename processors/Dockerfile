FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    pip install --no-cache-dir -r requirements.txt

COPY common/ /app/common
COPY services/ /app/services
COPY processors/ /app/processors
COPY shared_data /app/shared_data

CMD ["ray", "start", "--address=ray-head:6379", "--block"]
