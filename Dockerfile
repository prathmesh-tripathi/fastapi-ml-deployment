# Base Python image
FROM python:3.12-slim

# Working directory inside container
WORKDIR /app

# Copy dependency file first
COPY requirements.txt .

# Install exact dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and model
COPY app ./app
COPY models ./models

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]