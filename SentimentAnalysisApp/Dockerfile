# Use the official Python image as the base image
FROM python:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Update pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define environment variable
ENV NAME SentimentAnalysisApp

# Command to run the FastAPI application
CMD ["uvicorn", "Fapp:app", "--host", "0.0.0.0", "--port", "8000"]
