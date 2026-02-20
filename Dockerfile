# 1. Base image with Python
FROM public.ecr.aws/docker/library/python:3.10-slim

# 2. Install system dependencies (FFmpeg)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

# 3. Set working directory inside container
WORKDIR /app

# 4. Copy dependency list
COPY requirements.txt .

# 5. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy project files
COPY orchestrator/ orchestrator/
COPY services/ services/
COPY common/ common/
COPY processors/ processors/
COPY shared_data/ shared_data/

# 7. Run your orchestrator
CMD ["python", "-m", "orchestrator.main"]
