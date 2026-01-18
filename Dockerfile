# Use the official Python 3.9 image
FROM python:3.9-slim

# Set the working directory to the root of the project inside the container
WORKDIR /input

# Copy the entire project to the container
COPY . /input

# Install dependencies
# We look for requirements.txt in the Web_Application folder
RUN pip install --no-cache-dir -r Web_Application/requirements.txt

# Create the uploads directory and set permissions to allow writing
# Hugging Face Spaces run as a non-root user (ID 1000) by default
RUN mkdir -p Web_Application/static/uploads && \
    chmod 777 Web_Application/static/uploads

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Set the working directory to where the app is located to ensure relative imports work
WORKDIR /input/Web_Application

# Command to run the application using Gunicorn
# -b 0.0.0.0:7860 binds to all interfaces on port 7860
# app:app refers to the 'app' object in 'app.py'
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
