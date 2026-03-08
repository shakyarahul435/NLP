FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/backend

COPY app/backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/backend /app/backend
COPY a5_outputs /app/a5_outputs

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
