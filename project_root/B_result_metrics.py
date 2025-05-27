import argparse
import json
import unicodedata
from pathlib import Path
import logging
import re
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from bert_score import score

# python3 result_metrics3.py --language all --input-dir ./all_results_injson_format/ --output-dir ./Final_Results_injson_format

# --- Configuration ---
PROJECT_ROOT = Path('/home2/nagaraju.vuppala/ABHI/riddle2/project_root')
LANGUAGES = ["kannada", "malayalam", "telugu", "tamil", "hindi", "bengali", "gujarati"]
LANGUAGE_CODES = {
    "kannada": "kn", "malayalam": "ml", "telugu": "te",
    "tamil": "ta", "hindi": "hi", "bengali": "bn", "gujarati": "gu"
}
PROMPT_FILE_TYPES = [
    "zero_shot_template_test_from_original",
    "2_shot_context_recon_template_test_from_original",
    "4_shot_context_recon_template_test_from_original",
    "2_shot_method1_pattern_template_test_from_original",
    "4_shot_method1_pattern_template_test_from_original",
    "2_shot_sem_sim_recon_template_test_from_saba",
    "4_shot_sem_sim_recon_template_test_from_saba"
]
METHOD_NAMES = {
    "zero_shot_template_test_from_original": "0-shot_Standard",
    "2_shot_context_recon_template_test_from_original": "2-shot_RandomExemplars",
    "4_shot_context_recon_template_test_from_original": "4-shot_RandomExemplars",
    "2_shot_method1_pattern_template_test_from_original": "2-shot_RISCORE-like",
    "4_shot_method1_pattern_template_test_from_original": "4-shot_RISCORE-like",
    "2_shot_sem_sim_recon_template_test_from_saba": "2-shot_SemanticSimilarExemplars",
    "4_shot_sem_sim_recon_template_test_from_saba": "4-shot_SemanticSimilarExemplars"
}
EVAL_MODELS = [
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-04-17",
    "meta-llama_llama-4-maverick",
    "meta-llama_llama-4-scout",
    "mistralai_mistral-saba"
]

# --- Setup Logging ---
def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def normalize_text(text: str) -> str:
    """Normalize text for accuracy comparison: lowercase, strip whitespace, Unicode normalization."""
    if not text or text in ["GROUND_TRUTH_NOT_FOUND", "GENERATED_OUTPUT_NOT_FOUND", "PROMPT_NOT_FOUND"]:
        return ""
    text = unicodedata.normalize('NFKC', text).strip().lower()
    return re.sub(r'\s+', ' ', text)

def exact_match_text(text1: str, text2: str) -> bool:
    """Check for exact match without normalization (case-sensitive, preserves whitespace)."""
    if not text1 or not text2 or text1 in ["GENERATED_OUTPUT_NOT_FOUND"] or text2 in ["GROUND_TRUTH_NOT_FOUND"]:
        return False
    return text1 == text2

def calculate_accuracy(generated: str, ground_truth: str) -> bool:
    """Check if generated answer matches ground truth after normalization."""
    return normalize_text(generated) == normalize_text(ground_truth)

def calculate_bertscore(generated_answers: list, ground_truth_answers: list, lang_code: str) -> list:
    """Calculate BERTScore F1 for a list of generated and ground truth answers."""
    try:
        P, R, F1 = score(
            generated_answers,
            ground_truth_answers,
            lang=lang_code,
            model_type="bert-base-multilingual-cased",
            verbose=False
        )
        return F1.tolist()
    except Exception as e:
        logging.error(f"Error calculating BERTScore: {e}")
        return [0.0] * len(generated_answers)

def evaluate_language(input_dir, language, output_dir, metrics, method_filter):
    """Evaluate metrics for a single language."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Load consolidated data from input_dir
    consolidated_json = input_dir / f"{language}_consolidated_model_outputs.json"
    if not consolidated_json.exists():
        logging.error(f"Consolidated JSON not found: {consolidated_json}")
        return
    
    try:
        with open(consolidated_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading {consolidated_json}: {e}")
        return
    
    # Filter by method if specified
    lang_data = [entry for entry in data if not method_filter or entry["prompt_file_type_key"] == method_filter]
    if not lang_data:
        logging.warning(f"No data found for language: {language}" + (f" and method: {method_filter}" if method_filter else ""))
        return
    
    # Initialize result structure
    results = defaultdict(lambda: defaultdict(list))  # model -> method -> [scores]
    counts = defaultdict(lambda: defaultdict(int))    # model -> method -> count of valid entries
    
    # Process entries
    for entry in tqdm(lang_data, desc=f"Processing {language} entries"):
        model = entry["model_name"]
        method = entry["method_name"]
        generated = entry["generated_answer"]
        ground_truth = entry["ground_truth_answer"]
        prompt_file_key = entry["prompt_file_type_key"]
        
        # Skip invalid entries
        if not generated or not ground_truth or generated == "GENERATED_OUTPUT_NOT_FOUND" or ground_truth == "GROUND_TRUTH_NOT_FOUND":
            logging.debug(f"Skipping invalid entry for {model}/{method} (prompt: {prompt_file_key}): riddle_index={entry['riddle_index_in_file']}, generated='{generated[:50]}...', ground_truth='{ground_truth[:50]}...'")
            continue
        
        # Calculate metrics
        entry_scores = {"riddle_index": entry["riddle_index_in_file"]}
        if "accuracy" in metrics:
            accuracy = calculate_accuracy(generated, ground_truth)
            entry_scores["accuracy"] = 1.0 if accuracy else 0.0
        if "exactmatch" in metrics:
            exact_match = exact_match_text(generated, ground_truth)
            entry_scores["exactmatch"] = 1.0 if exact_match else 0.0
        results[model][method].append(entry_scores)
        counts[model][method] += 1
    
    # Calculate BERTScore if requested
    if "bertscore" in metrics:
        bertscore_inputs = defaultdict(lambda: defaultdict(list))  # model -> method -> [riddle_index, generated, ground_truth]
        for entry in lang_data:
            model = entry["model_name"]
            method = entry["method_name"]
            generated = entry["generated_answer"]
            ground_truth = entry["ground_truth_answer"]
            if not generated or not ground_truth or generated == "GENERATED_OUTPUT_NOT_FOUND" or ground_truth == "GROUND_TRUTH_NOT_FOUND":
                continue
            bertscore_inputs[model][method].append((entry["riddle_index_in_file"], generated, ground_truth))
        
        for model in bertscore_inputs:
            for method in bertscore_inputs[model]:
                riddle_indices, generated_list, ground_truth_list = zip(*bertscore_inputs[model][method])
                bert_f1_scores = calculate_bertscore(generated_list, ground_truth_list, LANGUAGE_CODES[language])
                for idx, f1 in zip(riddle_indices, bert_f1_scores):
                    for res in results[model][method]:
                        if res["riddle_index"] == idx:
                            res["bertscore_f1"] = f1
                            break
                    else:
                        results[model][method].append({"riddle_index": idx, "bertscore_f1": f1})
                        counts[model][method] += 1
    
    # Aggregate results
    aggregated_results = {
        "language": language.capitalize(),
        "metrics": ", ".join(metrics),
        "method_filter": method_filter if method_filter else "all",
        "results": {}
    }
    for model in results:
        aggregated_results["results"][model] = {}
        for method in results[model]:
            scores = results[model][method]
            valid_count = counts[model][method]
            if not valid_count:
                continue
            aggregated_results["results"][model][method] = {"valid_entries": valid_count}
            if "accuracy" in metrics:
                accuracy_scores = [s["accuracy"] for s in scores if "accuracy" in s]
                aggregated_results["results"][model][method]["average_accuracy"] = round(np.mean(accuracy_scores) * 100, 2) if accuracy_scores else 0.0
            if "exactmatch" in metrics:
                exactmatch_scores = [s["exactmatch"] for s in scores if "exactmatch" in s]
                aggregated_results["results"][model][method]["average_exactmatch"] = round(np.mean(exactmatch_scores) * 100, 2) if exactmatch_scores else 0.0
            if "bertscore" in metrics:
                bert_f1_scores = [s["bertscore_f1"] for s in scores if "bertscore_f1" in s]
                aggregated_results["results"][model][method]["average_bertscore_f1"] = round(np.mean(bert_f1_scores) * 100, 2) if bert_f1_scores else 0.0
    
    # Save results in evaluation_outputs folder
    # evaluation_dir = output_dir / "evaluation_outputs"
    evaluation_dir = output_dir
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    output_file = evaluation_dir / f"{language}_evaluation_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_results, f, ensure_ascii=False, indent=4)
        logging.info(f"Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to {output_file}: {e}")

def main(args):
    project_root = Path(args.project_root)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate input directory
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    
    valid_metrics = ["accuracy", "bertscore", "exactmatch", "all"]
    if args.metric not in valid_metrics:
        logging.error(f"Invalid metric: {args.metric}. Choose from {valid_metrics}")
        return
    metrics = ["accuracy", "bertscore", "exactmatch"] if args.metric == "all" else [args.metric]
    
    if args.method and args.method not in PROMPT_FILE_TYPES:
        logging.error(f"Invalid method: {args.method}. Choose from {PROMPT_FILE_TYPES}")
        return
    
    languages = LANGUAGES if args.language.lower() == 'all' else [args.language.lower()]
    
    for language in languages:
        if language not in LANGUAGES:
            logging.error(f"Invalid language: {language}. Choose from {LANGUAGES}")
            continue
        logging.info(f"Evaluating language: {language}")
        evaluate_language(input_dir, language, output_dir, metrics, args.method)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate riddle model outputs for accuracy, BERTScore, and exact match.")
    parser.add_argument("--project-root", default="/home2/nagaraju.vuppala/ABHI/riddle2/project_root", help="Project root directory")
    parser.add_argument("--input-dir", default=None, help="Directory containing consolidated JSON files (defaults to <project_root>/consolidated_outputs)")
    parser.add_argument("--language", default="all", help="Language to evaluate or 'all' (default: all)")
    parser.add_argument("--metric", choices=["accuracy", "bertscore", "exactmatch", "all"], default="all", help="Metric to compute: accuracy, bertscore, exactmatch, or all")
    parser.add_argument("--method", choices=PROMPT_FILE_TYPES, default=None, help="Specific method base filename to evaluate (optional)")
    parser.add_argument("--output-dir", default="/home2/nagaraju.vuppala/ABHI/riddle2/project_root", help="Directory for output JSON files")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    args = parser.parse_args()
    
    # Set default input_dir if not provided
    args.input_dir = args.input_dir or (Path(args.project_root) / "consolidated_outputs")
    
    setup_logging(args.verbose)
    main(args)