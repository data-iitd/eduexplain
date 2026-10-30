import copy
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from json_utils import parse_and_validate, count_gold_sets, get_gold_set


# ==============================================================================
# Paths
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = SCRIPT_DIR.parent

# Defaults (all overridable from the command line, see parse_args()).
PROMPT_PATH = REPO_ROOT / "prompts" / "storyboard_prompt.txt"
INPUT_PATH = REPO_ROOT / "data" / "gold_sets.json"
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "storyboards.json"


# ==============================================================================
# Model Configuration
# ==============================================================================

MODEL_PATH = REPO_ROOT / "models" / "Qwen3.5-9B-Instruct"

TEMPERATURE = 0.2

MAX_NEW_TOKENS = 4096


def parse_args():
    import argparse
    global PROMPT_PATH, INPUT_PATH, OUTPUT_PATH, OUTPUT_DIR, MODEL_PATH
    global TEMPERATURE, MAX_NEW_TOKENS
    ap = argparse.ArgumentParser(description="Generate storyboards from gold resource sets.")
    ap.add_argument("--input", type=Path, default=INPUT_PATH, help="Gold-set JSON (list of queries).")
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output storyboard JSON.")
    ap.add_argument("--prompt", type=Path, default=PROMPT_PATH, help="System prompt file.")
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Local HF model folder.")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    a = ap.parse_args()
    PROMPT_PATH, INPUT_PATH, OUTPUT_PATH, MODEL_PATH = a.prompt, a.input, a.output, a.model_path
    OUTPUT_DIR = OUTPUT_PATH.parent
    TEMPERATURE, MAX_NEW_TOKENS = a.temperature, a.max_new_tokens


# ==============================================================================
# Load Files
# ==============================================================================

def load_prompt() -> str:
    """
    Load the master storyboard prompt exactly as stored.
    """
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Prompt file not found:\n{PROMPT_PATH}"
        )

    with open(
        PROMPT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return file.read()


def load_gold_sets() -> list:
    """
    Load all query objects from the gold-set input file.

    Each query object may contain ONE OR MORE gold_sets. Every gold_set
    must be generated into its own, independent storyboard using that
    gold_set's complete resource collection.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Gold-set input file must contain a JSON list."
        )

    return data


# ==============================================================================
# Load Qwen Model
# ==============================================================================

def load_model():
    """
    Load the local Qwen instruct model and tokenizer.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "This script must be run on a GPU compute node."
        )

    print("Loading Qwen tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print("Loading Qwen model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )

    model.eval()

    print("Qwen model loaded successfully.")
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
    
    return tokenizer, model


# ==============================================================================
# Generation-unit expansion
# ==============================================================================

def build_generation_units(gold_sets_data: list) -> list:
    """
    Expand the list of query objects into a flat list of
    (query_item, gold_set_index, generation_id) generation units —
    one unit per gold_set, not one per query.

    A query with 2 gold_sets produces 2 generation units, each of
    which will be sent to the model separately and must produce its
    own independent storyboard.
    """

    units = []

    for query_item in gold_sets_data:

        query_id = query_item.get("query_id", "UNKNOWN")

        try:
            n_gold_sets = count_gold_sets(query_item)
        except ValueError as exc:
            print(
                f"WARNING: skipping query {query_id} "
                f"(invalid gold_sets): {exc}"
            )
            continue

        for gold_set_index in range(n_gold_sets):
            generation_id = f"{query_id}::gs{gold_set_index}"

            units.append(
                {
                    "query_item": query_item,
                    "gold_set_index": gold_set_index,
                    "generation_id": generation_id,
                }
            )

    return units


def build_model_input(
    query_item: dict,
    gold_set_index: int,
) -> dict:
    """
    Build the JSON payload sent to the model for ONE gold_set.

    The payload mirrors the full query_item schema (so the master
    prompt's declared input schema stays unchanged) but its "gold_sets"
    list contains ONLY the single gold_set being generated for. This
    guarantees the model always operates on exactly one gold_set per
    call, with no risk of it picking the wrong one or blending several.
    """

    gold_set = get_gold_set(query_item, gold_set_index=gold_set_index)

    single_gold_set_input = copy.deepcopy(query_item)
    single_gold_set_input["gold_sets"] = [gold_set]

    return single_gold_set_input


# ==============================================================================
# Output Handling
# ==============================================================================
def build_storyboard_id(
    query_id: str,
    gold_set_index: int,
) -> str:
    """
    Build a deterministic storyboard ID.

    Each gold_set belonging to the same query receives a unique
    storyboard ID based on its position in the gold_sets list.
    """

    return (
        f"SB_{query_id}_{gold_set_index + 1:03d}"
    )

def populate_alt_storyboard_ids(
    results: list,
) -> None:
    """
    Populate alt_storyboard_ids for every storyboard.

    Storyboards belonging to the same query are considered
    alternative storyboards of one another.

    A storyboard's own ID is excluded from its
    alt_storyboard_ids list.
    """

    storyboards_by_query = {}

    # --------------------------------------------------------------
    # Group all results by query_id
    # --------------------------------------------------------------

    for result in results:

        if not isinstance(result, dict):
            continue

        query_id = result.get("query_id")
        storyboard = result.get("storyboard")

        if not query_id or not isinstance(storyboard, dict):
            continue

        if query_id not in storyboards_by_query:
            storyboards_by_query[query_id] = []

        storyboards_by_query[query_id].append(
            storyboard
        )

    # --------------------------------------------------------------
    # Populate alternative storyboard IDs
    # --------------------------------------------------------------

    for query_id, storyboards in storyboards_by_query.items():

        storyboard_ids = [
            storyboard["storyboard_id"]
            for storyboard in storyboards
            if storyboard.get("storyboard_id")
        ]

        for storyboard in storyboards:

            current_storyboard_id = (
                storyboard.get("storyboard_id")
            )

            storyboard["alt_storyboard_ids"] = [
                storyboard_id
                for storyboard_id in storyboard_ids
                if storyboard_id != current_storyboard_id
            ]


def load_existing_results() -> list:
    """
    Load previously generated results.

    This allows the script to resume if it was interrupted.
    """

    if not OUTPUT_PATH.exists():
        return []

    with open(
        OUTPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Existing output file must contain a JSON list."
        )

    return data


def save_results(results: list) -> None:
    """
    Save all generated storyboards into one JSON file.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ==============================================================================
# Resume Handling
# ==============================================================================

def get_completed_generation_ids(results: list) -> set:
    """
    Return generation_ids that have already been successfully processed.

    IMPORTANT: keyed on generation_id (query_id + gold_set_index), NOT
    query_id alone. A query with multiple gold_sets must have every
    gold_set completed independently — finishing gold_set 0 must never
    cause gold_set 1 of the same query to be skipped.
    """

    completed_ids = set()

    for result in results:

        if not isinstance(result, dict):
            continue

        generation_id = result.get("generation_id")

        if generation_id:
            completed_ids.add(generation_id)

    return completed_ids


# ==============================================================================
# Qwen Generation
# ==============================================================================

def generate_storyboard(
    master_prompt: str,
    model_input: dict,
    tokenizer,
    model,
) -> str:
    """
    Generate a storyboard for ONE gold_set.

    The master prompt is passed as the system message.
    model_input (a query_item scoped to a single gold_set) is passed
    as the user message.
    """

    query_json = json.dumps(
        model_input,
        ensure_ascii=False,
        indent=2,
    )

    messages = [
        {
            "role": "system",
            "content": master_prompt,
        },
        {
            "role": "user",
            "content": query_json,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking = False,

    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[
        0,
        input_length:
    ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    response = response.strip()

    if not response:
        raise RuntimeError(
            "Qwen returned an empty response."
        )

    return response


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():

    parse_args()

    print("=" * 80)
    print("Set2Story Storyboard Generation")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Load prompt
    # --------------------------------------------------------------------------

    print("\nLoading master storyboard prompt...")

    master_prompt = load_prompt()

    print("Prompt loaded successfully.")

    # --------------------------------------------------------------------------
    # Load gold sets and expand into per-gold-set generation units
    # --------------------------------------------------------------------------

    print("\nLoading gold sets...")

    gold_sets_data = load_gold_sets()

    print(
        f"Loaded {len(gold_sets_data)} query objects."
    )

    generation_units = build_generation_units(gold_sets_data)

    n_queries = len(gold_sets_data)
    n_units = len(generation_units)

    print(
        f"Expanded into {n_units} generation unit(s) "
        f"(1 storyboard per gold_set) across {n_queries} quer(y/ies)."
    )

    # --------------------------------------------------------------------------
    # Load existing results
    # --------------------------------------------------------------------------

    existing_results = load_existing_results()

    completed_ids = get_completed_generation_ids(existing_results)

    # --------------------------------------------------------------------------
    # Normalize storyboard metadata for existing results
    # --------------------------------------------------------------------------

    for result in existing_results:

        if not isinstance(result, dict):
            continue

        query_id = result.get("query_id")
        gold_set_index = result.get("gold_set_index")
        storyboard = result.get("storyboard")

        if (
            query_id is None
            or gold_set_index is None
            or not isinstance(storyboard, dict)
        ):
            continue

        storyboard["storyboard_id"] = build_storyboard_id(
            query_id=query_id,
            gold_set_index=gold_set_index,
        )

    populate_alt_storyboard_ids(existing_results)

    print(
        f"Previously completed generation units: "
        f"{len(completed_ids)}"
    )

    # --------------------------------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------------------------------

    print("\nLoading Qwen model...")

    tokenizer, model = load_model()

    # --------------------------------------------------------------------------
    # Process generation units (one per gold_set)
    # --------------------------------------------------------------------------

    results = existing_results

    TEST_MODE = False
    TEST_MODE_NUM_QUERIES = 5

    if TEST_MODE:
        # Select ALL generation units belonging to the first N distinct
        # queries (in original file order), not the first N generation
        # units. A query with multiple gold_sets must have every one of
        # its gold_sets included together, or you'd silently test a
        # partial/incomplete query.
        seen_query_ids = []

        for unit in generation_units:
            qid = unit["query_item"].get("query_id", "UNKNOWN")
            if qid not in seen_query_ids:
                if len(seen_query_ids) >= TEST_MODE_NUM_QUERIES:
                    break
                seen_query_ids.append(qid)

        selected_query_ids = set(seen_query_ids)

        generation_units = [
            unit
            for unit in generation_units
            if unit["query_item"].get("query_id", "UNKNOWN")
            in selected_query_ids
        ]

        print("\nTEST MODE ENABLED")
        print(
            f"Processing first {len(seen_query_ids)} quer(y/ies) "
            f"({len(generation_units)} generation unit(s) total)."
        )

    total_units = len(generation_units)

    successful_count = len(completed_ids)

    failed_count = 0

    for index, unit in enumerate(generation_units, start=1):

        query_item = unit["query_item"]
        gold_set_index = unit["gold_set_index"]
        generation_id = unit["generation_id"]

        query_id = query_item.get("query_id", "UNKNOWN")

        # Skip already completed generation units
        if generation_id in completed_ids:

            print(
                f"\n[{index}/{total_units}] "
                f"Skipping {generation_id} "
                f"(already completed)"
            )

            continue

        print("\n" + "=" * 80)

        print(
            f"[{index}/{total_units}] "
            f"Processing generation unit: {generation_id}"
        )

        print("=" * 80)

        try:

            # --------------------------------------------------------------
            # Build the single-gold-set model input
            # --------------------------------------------------------------

            model_input = build_model_input(
                query_item=query_item,
                gold_set_index=gold_set_index,
            )

            # --------------------------------------------------------------
            # Generate storyboard
            # --------------------------------------------------------------

            raw_response = generate_storyboard(
                master_prompt=master_prompt,
                model_input=model_input,
                tokenizer=tokenizer,
                model=model,
            )

            safe_id = generation_id.replace(":", "_")
            debug_path = OUTPUT_DIR / f"{safe_id}_raw_response.txt"

            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )
            with open(
                debug_path,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(raw_response)

            # --------------------------------------------------------------
            # Parse and validate model output against THIS gold_set
            # --------------------------------------------------------------
            # model_input's gold_sets list contains only the single
            # gold_set for this unit, at index 0.

            storyboard = parse_and_validate(
                raw_response,
                model_input,
                gold_set_index=0,
            )
            storyboard["storyboard_id"] = build_storyboard_id(
                query_id=query_id,
                gold_set_index=gold_set_index,
            )
                
            storyboard["alt_storyboard_ids"] = []
            

            # --------------------------------------------------------------
            # Store result
            # --------------------------------------------------------------

            result = {
                "generation_id": generation_id,
                "query_id": query_id,
                "gold_set_index": gold_set_index,
                "storyboard": storyboard,
            }

            results.append(result)

            populate_alt_storyboard_ids(results)

            # --------------------------------------------------------------
            # Save immediately
            # --------------------------------------------------------------

            save_results(results)

            completed_ids.add(generation_id)

            successful_count += 1

            print(
                f"Successfully completed: "
                f"{generation_id}"
            )

            print(
                f"Progress: "
                f"{successful_count}/{total_units}"
            )

        except Exception as exc:

            failed_count += 1

            print(
                f"\nERROR processing "
                f"{generation_id}:"
            )

            print(exc)

            print(
                "Continuing to next generation unit..."
            )

            # Save completed work before continuing
            save_results(results)

            continue

    # --------------------------------------------------------------------------
    # Final Save
    # --------------------------------------------------------------------------
    
    populate_alt_storyboard_ids(results)

    save_results(results)

    print("\n" + "=" * 80)
    print("Storyboard generation completed.")
    print("=" * 80)

    print(
        f"Total generation units: {total_units}"
    )

    print(
        f"Successful: {successful_count}"
    )

    print(
        f"Failed in this run: {failed_count}"
    )

    print(
        f"\nOutput saved to:\n{OUTPUT_PATH}"
    )


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    main()