import os
import subprocess
from pathlib import Path

# --- CONFIG ---
PYTHON = "python"
SAMPLER_SCRIPT = "src/corpus_sampler.py"
AUDITOR_SCRIPT = "src/tokenization_auditor.py"

# Пути к enriched файлам
PATH_TAIGA = "data/interim/taiga_full_enriched.conllu"
PATH_SYNTAGRUS = "data/interim/syntagrus_full_enriched.conllu"

OUTPUT_DATASETS = "data/test_sets/tokenization"
OUTPUT_REPORTS = "reports/tokenization_audit"

# Модели
# MODELS = ["razdel", "mystem", "spacy", "slovnet", "deeppavlov", "stanza", "trankit", "udpipe"]
MODELS = ["razdel", "mystem", "spacy", "slovnet", "deeppavlov", "stanza", "udpipe"]

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def prepare_datasets():
    Path(OUTPUT_DATASETS).mkdir(parents=True, exist_ok=True)

    # 1. HARD LITERARY (Taiga, Fiction, Dialogues, Long)
    print("\n[1/3] Generating Hard Literary (Taiga)...")
    run_command([
        PYTHON, SAMPLER_SCRIPT,
        "--input", PATH_TAIGA,
        "--output", f"{OUTPUT_DATASETS}/hard_lit.conllu",
        "--query", "genre.str.contains('fiction', case=False, na=False) and is_dialogue == True and length > 19",
        "--limit", "500"
    ])

    # 2. MEDIUM LITERARY (Taiga, Fiction, Narrative, Medium length)
    print("\n[2/3] Generating Medium Literary (Taiga)...")
    run_command([
        PYTHON, SAMPLER_SCRIPT,
        "--input", PATH_TAIGA,
        "--output", f"{OUTPUT_DATASETS}/medium_lit.conllu",
        "--query",
        "genre.str.contains('fiction', case=False, na=False) and is_dialogue == False and length > 10 and length <= 25",
        "--limit", "500"
    ])

    # 3. MEDIUM GENERAL (SynTagRus, EXCLUDING Fiction)
    # Используем genre != 'fiction'
    print("\n[3/3] Generating Medium General (SynTagRus non-fiction)...")
    run_command([
        PYTHON, SAMPLER_SCRIPT,
        "--input", PATH_SYNTAGRUS,
        "--output", f"{OUTPUT_DATASETS}/medium_general.conllu",
        "--query", "genre != 'fiction' and length > 15",
        "--limit", "500"
    ])


def run_audits():
    datasets = list(Path(OUTPUT_DATASETS).glob("*.conllu"))
    if not datasets:
        print("Error: No datasets generated.")
        return

    print(f"\n[2/2] Starting Audit...")
    for ds in datasets:
        cmd = [
                  PYTHON, AUDITOR_SCRIPT,
                  "--input", str(ds),
                  "--output_dir", OUTPUT_REPORTS,
                  "--models"
              ] + MODELS

        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error auditing {ds.name}: {e}")


if __name__ == "__main__":
    prepare_datasets()
    run_audits()
    print("\nDONE. Run python analyze_audit_results.py")
