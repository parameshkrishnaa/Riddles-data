import os
import time
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
from google.cloud import aiplatform
from google.auth import credentials
from google.oauth2 import service_account
from google import genai
from google.genai import types
from openai import OpenAI

#python3 run_evaluation.py --model deepseek/deepseek-r1 --language Hindi --prompt-file zero_shot_template_test_from_original --retries 1

# --- Configuration ---
LANGUAGES = ["Kannada", "Malayalam", "Telugu", "Tamil", "Hindi", "Bengali", "Gujarati"]
LANGUAGE_CODES = {
    "Kannada": "kn",
    "Malayalam": "ml",
    "Telugu": "te",
    "Tamil": "ta",
    "Hindi": "hi",
    "Bengali": "bn",
    "Gujarati": "gu"
}
MODELS = [
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-04-17",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
    "qwen/qwen3-235b-a22b",
    "deepseek/deepseek-r1",
    "mistralai/mistral-saba",
    "sarvam-m"
]
PROMPT_FILES = [
    "zero_shot_template_test_from_original",
    "2_shot_context_recon_template_test_from_original",
    "4_shot_context_recon_template_test_from_original",
    "2_shot_method1_pattern_template_test_from_original",
    "4_shot_method1_pattern_template_test_from_original",
    "2_shot_sem_sim_recon_template_test_from_original",
    "4_shot_sem_sim_recon_template_test_from_original"
]
PROJECT_ROOT = Path("/home2/nagaraju.vuppala/ABHI/gemini/project_root")
SERVICE_ACCOUNT_KEY_PATH = "/home2/nagaraju.vuppala/ABHI/gemini/project_root/gemini-yourserviceaccount.json"  # Update with your path
PROJECT_ID = "gemini-account"  # Your Google Cloud project ID
LOCATION = "us-central1"  # Your Vertex AI location
OPENROUTER_API_KEY = "yourkey"  # Update with your OpenRouter API key
SARVAM_API_KEY = "yourkey"  # Update with your Sarvam API key
NUM_RETRIES = 2
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
]
GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    temperature=1,
    top_p=0.95,
    max_output_tokens=8192,
    safety_settings=SAFETY_SETTINGS,
)

# --- Setup Command-Line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for running riddle-solving models.")
    parser.add_argument("--model", choices=MODELS, required=True, help="Model to use for riddle solving.")
    parser.add_argument("--language", choices=LANGUAGES, required=True, help="Language of the riddles.")
    parser.add_argument("--prompt-file", choices=PROMPT_FILES, required=True, help="Prompt file to use.")
    parser.add_argument("--retries", type=int, default=NUM_RETRIES, help="Number of retries per riddle.")
    return parser.parse_args()

# --- Setup Logging ---
def setup_logging(log_file: Path):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logging.getLogger('').setLevel(logging.INFO)
    logging.getLogger('').addHandler(console_handler)
    logging.getLogger('').addHandler(file_handler)

# --- Setup Authentication for Gemini ---
def setup_gemini_authentication(key_path: str):
    try:
        creds = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
        aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
        logging.info(f"Successfully authenticated with project '{PROJECT_ID}' in location '{LOCATION}'.")
        return creds
    except Exception as e:
        logging.error(f"Error setting up Gemini authentication: {e}")
        return None

# --- Initialize Clients ---
def initialize_gemini_client(creds):
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
            credentials=creds
        )
        logging.info(f"Successfully initialized Gemini client.")
        return client
    except Exception as e:
        logging.error(f"Error initializing Gemini client: {e}")
        return None

def initialize_openrouter_client():
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        logging.info("Successfully initialized OpenRouter client.")
        return client
    except Exception as e:
        logging.error(f"Error initializing OpenRouter client: {e}")
        return None

def initialize_sarvam_client():
    try:
        client = OpenAI(
            base_url="https://api.sarvam.ai/v1",
            api_key=SARVAM_API_KEY,
        ).with_options(max_retries=1)
        logging.info("Successfully initialized Sarvam client.")
        return client
    except Exception as e:
        logging.error(f"Error initializing Sarvam client: {e}")
        return None

# --- Helper Functions ---
def get_prompt_file_path(language: str, prompt_file: str) -> Path:
    language_code = LANGUAGE_CODES[language]
    prompt_filename = f"{language_code}_{prompt_file}.txt"
    return PROJECT_ROOT / "prompts" / language.lower() / prompt_filename

def split_prompts_from_file(file_path: Path, language: str) -> list[str]:
    try:
        content = file_path.read_text(encoding='utf-8')
        parts = content.split(f"\n\nYou are an expert in solving {language} riddles.")
        prompts = [part.strip() for part in parts if part.strip()]
        logging.info(f"Read {len(prompts)} prompts from {file_path.name}")
        return prompts
    except FileNotFoundError:
        logging.error(f"Error: Prompt file not found at {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading or splitting file {file_path}: {e}")
        return []

def generate_with_gemini(client: genai.Client, prompt_text: str, model_name: str) -> str:
    if not client:
        logging.error("Error: Gemini client not initialized.")
        return "Error: Client not initialized."
    try:
        contents = [types.Content(role="user", parts=[types.Part(text=prompt_text)])]
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=GENERATE_CONTENT_CONFIG,
        )
        full_response = "".join(chunk.text for chunk in response_stream)
        return full_response.strip()
    except Exception as e:
        logging.error(f"An error occurred during Gemini API call: {e}")
        time.sleep(5)
        return f"Error generating response: {e}"

def generate_with_openrouter(client: OpenAI, prompt_text: str, model: str) -> str:
    if not client:
        logging.error("Error: OpenRouter client not initialized.")
        return "Error: Client not initialized."
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Riddle Pipeline",
            },
            extra_body={},
            model=model,
            messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"An error occurred during OpenRouter API call for {model}: {e}")
        time.sleep(5)
        return f"Error generating response: {e}"

def generate_with_sarvam(client: OpenAI, prompt_text: str, language: str) -> str:
    if not client:
        logging.error("Error: Sarvam client not initialized.")
        return "Error: Client not initialized."
    try:
        messages = [
            {"role": "system", "content": f"Respond only in {language}."},
            {"role": "user", "content": prompt_text},
        ]
        response = client.chat.completions.create(
            model="sarvam-m",
            messages=messages,
            reasoning_effort="medium",
            max_completion_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"An error occurred during Sarvam API call: {e}")
        time.sleep(5)
        return f"Error generating response: {e}"

# --- Main Pipeline ---
def main():
    args = parse_args()
    MODEL_NAME = args.model
    LANGUAGE = args.language
    PROMPT_FILE = args.prompt_file
    NUM_RETRIES = args.retries

    # Sanitize model name for file paths (replace '/' with '_')
    safe_model_name = MODEL_NAME.replace('/', '_')

    # Set up output and log directories
    output_dir = PROJECT_ROOT / "results" / LANGUAGE.lower() / safe_model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{LANGUAGE.lower()}_{PROMPT_FILE}_output.txt"
    log_file = output_dir / f"{LANGUAGE.lower()}_{safe_model_name}_log.txt"

    # Setup logging (after directory creation)
    setup_logging(log_file)
    logging.info(f"Starting pipeline for model: {MODEL_NAME}, language: {LANGUAGE}, prompt: {PROMPT_FILE}")

    # Initialize client based on model
    client = None
    if MODEL_NAME.startswith("gemini"):
        creds = setup_gemini_authentication(SERVICE_ACCOUNT_KEY_PATH)
        if creds:
            client = initialize_gemini_client(creds)
    elif MODEL_NAME == "sarvam-m":
        client = initialize_sarvam_client()
    else:
        client = initialize_openrouter_client()

    if not client:
        logging.error("Failed to initialize client. Exiting.")
        exit(1)

    # Read prompts
    prompt_file_path = get_prompt_file_path(LANGUAGE, PROMPT_FILE)
    prompts = split_prompts_from_file(prompt_file_path, LANGUAGE)
    if not prompts:
        logging.error("No prompts found. Exiting.")
        exit(1)

    # Generate responses
    try:
        with open(output_file, 'a', encoding='utf-8') as outfile:
            outfile.write(f"\n--- Starting New Generation Run at {time.ctime()} ---\n")
            outfile.write(f"Model: {MODEL_NAME}, Language: {LANGUAGE}, Prompt: {PROMPT_FILE}\n")
            outfile.write(f"Retries: {NUM_RETRIES}\n")
            outfile.write("-" * 50 + "\n\n")
            outfile.flush()

            for riddle_index, prompt_text in enumerate(tqdm(prompts, desc="Processing Riddles", unit="riddle")):
                logging.info(f"Processing Riddle #{riddle_index + 1}...")
                outfile.write(f"--- Riddle Index: {riddle_index} ---\n\n")
                outfile.write(f"Prompt:\n{prompt_text}\n\n")
                outfile.flush()

                for attempt in tqdm(range(1, NUM_RETRIES + 1), desc=f"Attempts for Riddle #{riddle_index + 1}", unit="attempt", leave=False):
                    logging.info(f"Attempt {attempt}/{NUM_RETRIES}...")
                    outfile.write(f"-- Attempt {attempt} --\n\n")

                    if MODEL_NAME.startswith("gemini"):
                        generated_output = generate_with_gemini(client, prompt_text, MODEL_NAME)
                    elif MODEL_NAME == "sarvam-m":
                        generated_output = generate_with_sarvam(client, prompt_text, LANGUAGE)
                    else:
                        generated_output = generate_with_openrouter(client, prompt_text, MODEL_NAME)

                    outfile.write("Generated Output:\n")
                    outfile.write(generated_output)
                    outfile.write("\n\n" + "-" * 30 + "\n\n")
                    outfile.flush()

                    logging.info(f"Generation complete for attempt {attempt}.")
                    time.sleep(2)

                outfile.write(f"--- End of Processing for Riddle Index: {riddle_index} ---\n\n")
                outfile.flush()

            outfile.write(f"\n--- Generation Run Finished at {time.ctime()} ---\n")
            outfile.flush()
        logging.info(f"Successfully completed all generations. Output saved to {output_file}")

    except IOError as e:
        logging.error(f"Error writing to output file {output_file}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()