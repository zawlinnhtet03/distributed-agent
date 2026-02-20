FROM public.ecr.aws/docker/library/python:3.10-slim

WORKDIR /app

COPY router/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY router /app/router

ENV PYTHONPATH=/app

CMD ["uvicorn", "router.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
