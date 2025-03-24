FROM python:3.12.3-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Expose the TCP socket port
EXPOSE 5505

# Create a startup script
RUN echo '#!/bin/bash\n\
# Start the watchdog in the background\n\
python watchdog.py & \n\
\n\
# Start the server in the foreground\n\
exec python server.py' > /app/start.sh && \
    chmod +x /app/start.sh

# Command to run both server and watchdog
CMD ["/app/start.sh"]