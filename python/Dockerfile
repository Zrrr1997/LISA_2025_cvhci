FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Install build tools
RUN apt-get update && \
    apt-get install -y build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/user

# Copy only the dependency files first
COPY requirements.txt ./


# Install dependencies (cached unless these files change)
RUN pip install --user monailabel==0.8.5 --no-cache-dir --break-system-packages
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Now copy the rest of your project
COPY . .

# Set the command to run the script
ENTRYPOINT ["python", "run_model.py"]

