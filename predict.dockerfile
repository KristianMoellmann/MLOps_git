# Base image
FROM python:3.8-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential parts from our computer to the container
COPY requirements.txt requirements.txt
COPY models/ models/
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY reports/ reports/

# Set working directory and install requirements
WORKDIR /
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# Make prediction script the entrypoint of the image (application to be run when the image is being executed)
ENTRYPOINT ["python", "-u", "src/models/predict_model.py", "evaluate"]
