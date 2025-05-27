import argparse
import json
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
PROJECT_ROOT = Path('/home2/nagaraju.vuppala/ABHI/riddle2/project_root')
LANGUAGES = ["kannada", "malayalam", "telugu", "tamil", "hindi", "bengali", "gujarati"]
METRICS = ["accuracy", "exactmatch", "bertscore"]
METRIC_LABELS = {
    "accuracy": "Average Accuracy (%)",
    "exactmatch": "Average Exact Match (%)",
    "bertscore": "Average BERTScore F1 (%)"
}
MODELS = [
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-04-17",
    "meta-llama_llama-4-maverick",
    "meta-llama_llama-4-scout",
    "mistralai_mistral-saba"
]
METHODS = [
    "0-shot_Standard",
    "2-shot_RandomExemplars",
    "4-shot_RandomExemplars",
    "2-shot_RISCORE-like",
    "4-shot_RISCORE-like",
    "2-shot_SemanticSimilarExemplars",
    "4-shot_SemanticSimilarExemplars"
]
MODEL_COLORS = {
    "gemini-2.5-pro-preview-05-06": "#1f77b4",  # Blue
    "gemini-2.5-flash-preview-04-17": "#ff7f0e",  # Orange
    "meta-llama_llama-4-maverick": "#2ca02c",    # Green
    "meta-llama_llama-4-scout": "#d62728",       # Red
    "mistralai_mistral-saba": "#9467bd"          # Purple
}

# --- Setup Logging ---
def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def load_evaluation_results(file_path):
    """Load evaluation results from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.debug(f"Loaded results from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def create_charts(input_dir, language, output_dir, metrics):
    """Create bar charts for specified language and metrics, saving in language-specific subfolders."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) / language  # Language-specific subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_file = input_dir / f"{language}_evaluation_results.json"
    data = load_evaluation_results(json_file)
    if not data:
        return
    
    results = data.get("results", {})
    if not results:
        logging.warning(f"No results found in {json_file}")
        return
    
    # Prepare data for plotting
    for metric in metrics:
        metric_key = f"average_{metric}" if metric != "bertscore" else "average_bertscore_f1"
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bar_width = 0.15
        index = np.arange(len(METHODS))
        for i, model in enumerate(MODELS):
            values = []
            for method in METHODS:
                try:
                    value = results.get(model, {}).get(method, {}).get(metric_key, 0.0)
                    values.append(value)
                except KeyError:
                    values.append(0.0)
            ax.bar(index + i * bar_width, values, bar_width, label=model, color=MODEL_COLORS[model])
        
        # Customize plot
        ax.set_xlabel("Methods")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{language.capitalize()} - {METRIC_LABELS[metric]}")
        ax.set_xticks(index + bar_width * (len(MODELS) - 1) / 2)
        ax.set_xticklabels(METHODS, rotation=45, ha="right")
        ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Tight layout to prevent clipping
        plt.tight_layout()
        
        # Save as PNG and SVG
        for fmt in ["PNG", "SVG"]:
            output_file = output_dir / f"{language}_{metric}.{fmt.lower()}"
            plt.savefig(output_file, format=fmt.lower(), bbox_inches="tight")
            logging.info(f"Saved {fmt} chart to {output_file}")
        
        plt.close()

def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    
    valid_metrics = ["accuracy", "exactmatch", "bertscore", "all"]
    if args.metrics not in valid_metrics:
        logging.error(f"Invalid metric: {args.metrics}. Choose from {valid_metrics}")
        return
    metrics = METRICS if args.metrics == "all" else [args.metrics.lower()]
    
    languages = LANGUAGES if args.language.lower() == "all" else [args.language.lower()]
    
    for language in languages:
        if language not in LANGUAGES:
            logging.error(f"Invalid language: {language}. Choose from {LANGUAGES}")
            continue
        logging.info(f"Generating charts for {language}")
        create_charts(input_dir, language, output_dir, metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize evaluation results as bar charts in PNG and SVG.")
    parser.add_argument("--project-root", default="/home2/nagaraju.vuppala/ABHI/riddle2/project_root", help="Project root directory (used for default input_dir)")
    parser.add_argument("--input-dir", default=None, help="Directory containing evaluation JSON files (default: <project_root>/evaluation_outputs)")
    parser.add_argument("--language", default="all", help="Language to visualize or 'all' (default: all)")
    parser.add_argument("--metrics", choices=["accuracy", "exactmatch", "bertscore", "all"], default="all", help="Metric to plot: accuracy, exactmatch, bertscore, or all")
    parser.add_argument("--output-dir", default="/home2/nagaraju.vuppala/ABHI/riddle2/project_root", help="Base directory for output charts")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    args = parser.parse_args()
    
    # Set default input_dir if not provided
    # args.input_dir = args.input_dir or (Path(args.project_root) / "evaluation_outputs")
    args.input_dir = args.input_dir or (Path(args.project_root))
    
    setup_logging(args.verbose)
    main(args)