import os
import time
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
import subprocess
import json
import requests
import openai

def get_available_models():
   
    # Initialize models
    models = []

     # Check OPENAI_API_KEY and add models
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        models = ["gpt-4o","gpt-3.5-turbo"]

    # Try to get ollama models
    try:
         # Use environment variable for base_url, fallback to localhost if not set
        base_url = os.getenv('OLLAMA_API_BASE_URL', 'http://localhost:11434')
        print(f"Using {base_url}")

        # Make HTTP GET request to fetch data from the API
        response = requests.get(f"{base_url}/api/tags")
        response.raise_for_status()  # Raise HTTPError for non-200 status codes

        # Parse the response JSON and extract ollama model names
        data = response.json()
        models.extend(model.get('name',[]) for model in data.get('models', []))

    except requests.exceptions.RequestException as e:
        print(f"Error during the API request: {e}")
    except Exception as e:
        print(f"Error extracting the Ollama models from the response: {e}")
    
    return models


def get_llm_output(llm_model, input_string):
    print(f"using {llm_model}")

    if "mixtral" in llm_model or "ollama" in llm_model:
        result = execute_ollama_script(llm_model, input_string)
    elif "gpt-3.5" in llm_model or "gpt-4" in llm_model:
        result = execute_openai_chatgpt(llm_model, input_string)
    else:
        # Raise an error if the model is not supported
        raise ValueError(f"Unsupported LLM model: {llm_model}")

    print(result)
    return result

def get_output_claude3(astring):
    return('get_output_claude3 not implemented')

def execute_ollama_script(llm_model, input_string):
    # Use environment variable for base_url, fallback to localhost if not set
    base_url = os.getenv('OLLAMA_API_BASE_URL', 'http://localhost:11434')
    
    print(f"using {base_url}")
    print(input_string)

    llm = Ollama(base_url=base_url, model=llm_model) 
    answer = llm.invoke(input_string)

    return answer
    
def execute_openai_chatgpt(llm_model,input_string):
    # Get API key from environment variable
    try:
        client = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
        assert client.api_key
    except:
        raise ValueError("OpenAI API key is missing. Set the 'OPENAI_API_KEY' environment variable.")

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "user", "content": input_string}
            ],
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        return answer
    
    except (KeyError, IndexError, ValueError) as e:
        # Handle specific issues with the response structure
        print(f"Error with OpenAI response: {e}")
        # return "Sorry, there was an error with the response."
        raise
    
    except Exception as e:
        # Handle other unexpected errors
        print(f"An unexpected error occurred: {e}")
        # return "Sorry, an unexpected error occurred."
        raise
    
    return answer
