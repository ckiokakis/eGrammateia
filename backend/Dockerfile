FROM python:3.10-slim

# Set environment
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_compact.txt .
RUN pip install --no-cache-dir -r requirements_compact.txt

# Copy app
COPY . .

# Run the script
CMD ["python", "app.py"]
