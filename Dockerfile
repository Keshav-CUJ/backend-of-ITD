# Use official Python image
FROM python:3.9

# Set working directory in container
WORKDIR /app

# Copy only requirements first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the uploads folder exists
RUN mkdir -p /app/uploads

# Copy the rest of the application code
COPY . .

# Expose the Flask port (change if needed)
EXPOSE 5000

# Run Flask server
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

