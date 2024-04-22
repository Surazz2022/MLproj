# Use the official Python image with version 3.12 as a base
FROM python:3.12-slim AS builder

# Install dependencies for building and testing
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Start a new stage
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code from the previous stage
COPY --from=builder /app .

# Specify the command to run your application
CMD ["uvicorn", "Fapp:app", "--host", "0.0.0.0", "--port", "8000"]
