
# Docker Installation
### 0. Install Docker if not already done

### 1. Update API Key for chatGPT
In the `docker-compose.yml` file:
- Change the `OPENAI_API_KEY` if you are using OpenAI ChatGPT.

### 2. Access Local Hard Drive
For accessing your local hard drive for data, modify the `volumes` section:
- From:
  ```yaml
  volumes: 
    device: /home/mamat/accplatform/article1-sanofi/data
  ```
- To:
  ```yaml
  volumes:
    device: [location on your hard drive]
  ```

### 3. Build and Run Docker
```bash
docker compose build
docker compose up
```

### 4. (Optional) install a new Ollama model
Open your browser and use http://{docker_container_machine_ip}:3000/ to install a new model.

### 5. Access the app
Open your browser and use http://{docker_container_machine_ip}:8501/ with {docker_container_machine_ip} being localhost most of the time

"Enter the directory containing CSV files:": /home/streamlit/data/Kras


# Manual Installation on Linux
### 0. Install python, conda

### 1. Configure Environment Variable
Set the OpenAI API key:
```bash
export OPENAI_API_KEY=enter_your_chatgpt_api_key_here
```

### 2. Install and Run Python Environment
Execute the following script to install and run the Python environment:
```bash
./install.sh
```

### 3. Install and Run Ollama on Linux
Follow the installation instructions for Ollama from [https://ollama.com/](https://ollama.com/).

To install a model, follow the instructions provided on the website.

If necessary, update the environment variable for Ollama:
```bash
export OLLAMA_API_BASE_URL=http://localhost:11434
```

### 4. Launch the Streamlit App
Activate the `evocell` environment and launch the app:
```bash
conda activate evocell
cd evocell/app
streamlit run main.py
```

# Manual Installation on Windows
### 0. Install python, conda

### 1. Configure Environment Variable
To set the OpenAI API key, open a Command Prompt and run:
```bash
set OPENAI_API_KEY=enter_your_chatgpt_api_key_here
```

### 2. Install and Run Python Environment
Run the `install.bat` file to install dependencies:
```bash
install.bat
```

### 3. Install and Run Ollama on Windows
Follow the instructions for installing Ollama from [https://ollama.com/](https://ollama.com/).

After installation, update the environment variable if necessary:
```bash
set OLLAMA_API_BASE_URL=http://localhost:11434
```

### 4. Launch the Streamlit App
Activate the Python environment and run the Streamlit app:
```bash
conda activate evocell
cd evocell
streamlit run evocell_app.py
```

# Manual Installation on macOS
### 0. Install python, conda

### 1. Configure Environment Variable
Set the OpenAI API key in the terminal:
```bash
export OPENAI_API_KEY=enter_your_chatgpt_api_key_here
```

### 2. Install and Run Python Environment
Run the `install.sh` script to install dependencies:
```bash
./install.sh
```

### 3. Install and Run Ollama on macOS
Follow the installation instructions for Ollama from [https://ollama.com/](https://ollama.com/).

After installation, if necessary, update the environment variable:
```bash
export OLLAMA_API_BASE_URL=http://localhost:11434
```

### 4. Launch the Streamlit App
Activate the Python environment and run the app:
```bash
conda activate evocell
cd evocell
streamlit run evocell_app.py
```
