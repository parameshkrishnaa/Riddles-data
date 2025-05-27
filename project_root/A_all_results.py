import argparse
import json
import re
from pathlib import Path
import logging

#python3 all_results4.py --language malayalam --output-dir ./all_results_injson_format
#python3 all_results4.py --language all --output-dir ./all_results_injson_format

# --- Configuration ---
PROJECT_ROOT = Path('/home2/nagaraju.vuppala/ABHI/riddle2/project_root')
LANGUAGES = ["kannada", "malayalam", "telugu", "tamil", "hindi", "bengali", "gujarati"]
METHOD_MAPPING = {
    "zero_shot_template_test_from_original": "0-shot_Standard",
    "2_shot_context_recon_template_test_from_original": "2-shot_RandomExemplars",
    "4_shot_context_recon_template_test_from_original": "4-shot_RandomExemplars",
    "2_shot_method1_pattern_template_test_from_original": "2-shot_RISCORE-like",
    "4_shot_method1_pattern_template_test_from_original": "4-shot_RISCORE-like",
    "2_shot_sem_sim_recon_template_test_from_original": "2-shot_SemanticSimilarExemplars",
    "4_shot_sem_sim_recon_template_test_from_original": "4-shot_SemanticSimilarExemplars"
}

# --- Setup Logging ---
def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def load_riddles(file_path):
    """Load riddles from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        riddles = data.get('riddles', [])
        logging.debug(f"Loaded {len(riddles)} riddles from {file_path}")
        return riddles
    except Exception as e:
        logging.error(f"Error loading riddles from {file_path}: {e}")
        return []

def parse_model_output(file_path):
    """Parse model output file to extract riddle index and generated answer."""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            logging.warning(f"Empty file: {file_path}")
            return results
        
        riddle_sections = re.split(r'--- Riddle Index: (\d+) ---', content)[1:]
        if len(riddle_sections) < 2:
            logging.warning(f"No riddle blocks found in {file_path}. Content start: {content[:200]}...")
            return results

        for i in range(0, len(riddle_sections), 2):
            index = int(riddle_sections[i])
            section = riddle_sections[i + 1]
            
            prompt_match = re.search(r'Prompt:\s*([\s\S]*?)\n\n-- Attempt \d+ --', section, re.MULTILINE)
            full_prompt = prompt_match.group(1).strip() if prompt_match else "PROMPT_NOT_FOUND"
            
            answer_match = re.search(r'Generated Output:\s*([\s\S]*?)(?=\n\n-{20,}\n\n|\Z)', section, re.MULTILINE)
            generated_answer = answer_match.group(1).strip() if answer_match else "GENERATED_OUTPUT_NOT_FOUND"
            
            results.append({
                'riddle_index_in_file': index,
                'full_prompt_to_model': full_prompt,
                'generated_answer': generated_answer
            })
        logging.debug(f"Parsed {len(results)} entries from {file_path}")
    except Exception as e:
        logging.error(f"Error parsing model output {file_path}: {e}")
    return results

def map_method_name(prompt_file_type_key):
    """Map prompt file type key to method name."""
    method_name = METHOD_MAPPING.get(prompt_file_type_key, "Unknown")
    if method_name == "Unknown":
        logging.warning(f"Unknown prompt file type key: {prompt_file_type_key}")
    return method_name

def generate_output_json(project_root, language, output_dir):
    """Generate JSON output for a single language, saving in consolidated_outputs folder."""
    output = []
    project_root = Path(project_root)
    riddle_file = project_root / "data" / "best4_riddles" / f"{language}_best_riddles.json"
    results_dir = project_root / "results" / language
    
    # Load riddles
    riddles = load_riddles(riddle_file)
    if not riddles:
        logging.error(f"No riddles found in {riddle_file}")
        return output
    
    # Process each model directory
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        
        # Process each output file
        for output_file in model_dir.glob("*_output.txt"):
            filename = output_file.stem
            if not filename.startswith(f"{language}_"):
                logging.warning(f"Skipping unexpected file: {output_file}")
                continue
            prompt_file_type_key = filename[len(f"{language}_"):-len("_output")]
            
            method_name = map_method_name(prompt_file_type_key)
            
            model_outputs = parse_model_output(output_file)
            
            for model_output in model_outputs:
                index = model_output['riddle_index_in_file']
                # Adjust for 0-based indexing if necessary
                if index < len(riddles):
                    riddle = riddles[index]
                    output.append({
                        "language": language.capitalize(),
                        "model_name": model_name,
                        "method_name": method_name,
                        "prompt_file_type_key": prompt_file_type_key,
                        "riddle_index_in_file": index,
                        "original_riddle": riddle['original_riddle'],
                        "ground_truth_answer": riddle['original_answer'],
                        "full_prompt_to_model": model_output['full_prompt_to_model'],
                        "generated_answer": model_output['generated_answer']
                    })
                else:
                    logging.warning(f"Riddle index {index} out of range for {output_file}")
    
    # Save output in consolidated_outputs folder
    # consolidated_dir = output_dir / "consolidated_outputs"
    consolidated_dir = output_dir
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    output_file = consolidated_dir / f"{language}_consolidated_model_outputs.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved {len(output)} entries to {output_file}")
    except Exception as e:
        logging.error(f"Error saving output to {output_file}: {e}")
    
    return output

def main(args):
    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    languages = LANGUAGES if args.language.lower() == 'all' else [args.language.lower()]
    
    for language in languages:
        if language not in LANGUAGES:
            logging.error(f"Invalid language: {language}. Choose from {LANGUAGES}")
            continue
        logging.info(f"Processing language: {language}")
        generate_output_json(project_root, language, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate riddle model outputs for one or all languages.")
    parser.add_argument("--project-root", default="/home2/nagaraju.vuppala/ABHI/riddle2/project_root", help="Project root directory")
    parser.add_argument("--language", default="all", help="Language to process or 'all' (default: all)")
    parser.add_argument("--output-dir", default="/home2/nagaraju.vuppala/ABHI/riddle2/project_root", help="Directory for output JSON files")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    main(args)