from pathlib import Path

# ============================================================
# Model Configuration
# ============================================================

MODEL_NAME = "~/models/Qwen2.5-7B-Instruct" #attach model path here

TEMPERATURE = 0.7

MAX_RETRIES = 3

TOP_P=0.9

REPETITION_PENALTY = 1.1

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_JSON = SCRIPT_DIR/"query_generation_bundles.json"

OUTPUT_JSON = SCRIPT_DIR/"generated_queries_hierarchial_new.json"

SYSTEM_PROMPT = SCRIPT_DIR/"system_prompt.txt"

PROMPT_FILES = {

    "single": SCRIPT_DIR/"single_prompt.txt",

    "pair": SCRIPT_DIR/"pair_prompt.txt",

    "triple": SCRIPT_DIR/"triple_prompt.txt"
}

# ==========================================================
# Retrieval Configuration
# ==========================================================

CONCEPT_LIMIT = 20

BM25_K = 10

DENSE_K = 10

MAX_CANDIDATES = 30

EMBEDDING_MODEL = "~/models/bge-base-en-v1.5" #attach model path here
