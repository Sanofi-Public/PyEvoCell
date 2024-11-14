# Check if Ollama is already installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Installing now..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama is already installed."
fi

ollama serve

# Check if the model mixtral:8x7b is already pulled
if ! ollama list | grep -q "mixtral:8x7b"; then
    echo "Model mixtral:8x7b not found. Pulling model..."
    ollama pull mixtral:8x7b
else
    echo "Model mixtral:8x7b is already pulled."
fi

# Check if the model mixtral:8x22b is already pulled
if ! ollama list | grep -q "mixtral:8x22b"; then
    echo "Model mixtral:8x22b not found. Pulling model..."
    ollama pull mixtral:8x22b
else
    echo "Model mixtral:8x22b is already pulled."
fi

/bin/ollama serve &
