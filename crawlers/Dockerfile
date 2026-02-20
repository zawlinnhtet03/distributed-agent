FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

RUN pip install celery redis playwright

COPY crawlers/ /app/crawlers
COPY shared_data /app/shared_data

CMD ["python", "crawlers/scraper.py"]
