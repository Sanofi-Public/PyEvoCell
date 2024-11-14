#!/bin/sh

# Change ownership of the mounted directory
chown -R ollama:ollama /home/ollama/.ollama

# Switch to ollama user and execute the main process
su -s /bin/sh ollama <<'EOF'
/bin/ollama serve &

sleep 5
#curl -X POST http://ollama:11434/api/pull -d '{"name": "llama2"}'
#sleep 10
tail -f /dev/null
EOF