FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && \
    apt-get install build-essential curl unzip file git ruby-full locales --no-install-recommends -y && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install -r requirements.txt

# Expose port 8502 (Streamlit's default port)
EXPOSE 8502

# Start the Streamlit app when the container runs
CMD ["streamlit", "run", "app.py", "--server.port", "8502", "--server.headless", "true", "--server.enableCORS", "false"]
