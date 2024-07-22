# Use the official Python 3.10.0 image from the Docker Hub
FROM python:3.10.0-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirement.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Expose port 8000 for the FastAPI application (or adjust as needed)
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
