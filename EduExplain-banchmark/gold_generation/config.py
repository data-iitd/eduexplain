# ==========================================================
# Model Configuration
# ==========================================================

# MODEL_NAME = "Qwen/Qwen2.5-7B Instruct"
MODEL_NAME = "~/models/Qwen2.5-7B-Instruct"

TEMPERATURE = 0.5

TOP_P = 0.5

MAX_NEW_TOKENS = 8192

MAX_RETRIES = 3

# ==========================================================
# File Paths
# ==========================================================
INPUT_JSON ="./gold_generation/query_candidate_pool_2_new.json"

OUTPUT_JSON = "gold_sets_new.json"

SYSTEM_PROMPT = "system_prompt.txt"

GOLD_PROMPT = "gold_generation_prompt.txt"
