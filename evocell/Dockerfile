# Use an official Python image as the base
FROM python:3.12-slim

# Set environment variables to avoid prompts during install and configure the app directory
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install dependencies for conda and system tools needed for EvoCell and Python
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \ 
    rm /tmp/miniconda.sh

# Add conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Create EvoCell conda environment
RUN conda create -n evocell python=3.10 -y && \
    conda clean -afy

# Copy the requirements.txt file
COPY requirements.txt .

# Activate the conda environment explicitly and install the dependencies
# Use a single RUN instruction for conda activate and subsequent commands
RUN /bin/bash -c "source activate evocell && \
    pip install -r requirements.txt"

# Set up the working directory
WORKDIR /home/streamlit

# Copy the application code into the container
COPY ./app /home/streamlit
COPY ./www  /home/streamlit/www

# Expose the default Streamlit port
EXPOSE 8501

# Ensure the conda environment is activated before running streamlit
ENTRYPOINT ["/bin/bash", "-c", "source activate evocell && streamlit run main.py"]
#CMD ["tail", "-f", "/dev/null"]

