import json
import re
import time
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from config import *

# ==========================================================
# Load Prompts
# ==========================================================

def load_prompt(path):

    with open(path, "r", encoding="utf-8") as f:

        return f.read()


SYSTEM_PROMPT = load_prompt(SYSTEM_PROMPT)

GOLD_PROMPT = load_prompt(GOLD_PROMPT)

# ==========================================================
# Load Model
# ==========================================================

print("=" * 70)
print("Loading Model")
print("=" * 70)

print(f"Model : {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

generator = pipeline(

    "text-generation",

    model=model,

    tokenizer=tokenizer

)

print("Model Loaded.\n")

# ==========================================================
# Generation
# ==========================================================

def generate(prompt):

    messages = [

        {
            "role": "system",

            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",

            "content": prompt
        }

    ]

    output = generator(

        messages,

        max_new_tokens=MAX_NEW_TOKENS,

        temperature=TEMPERATURE,

        top_p=TOP_P,

        do_sample=True,

        return_full_text=False

    )

    return output[0]["generated_text"]

# ==========================================================
# JSON Extraction
# ==========================================================

def extract_json(text):

    """
    Extract JSON object from model output.
    """

    if text is None:

        return None

    if isinstance(text, dict):

        return text

    if isinstance(text, list):

        return text

    text = text.strip()

    # Remove markdown fences

    text = re.sub(

        r"^```(?:json)?",

        "",

        text,

        flags=re.MULTILINE

    )

    text = re.sub(

        r"```$",

        "",

        text,

        flags=re.MULTILINE

    )

    text = text.strip()

    try:

        return json.loads(text)

    except:

        pass

    start = text.find("{")

    end = text.rfind("}")

    if start == -1 or end == -1:

        return None

    try:

        return json.loads(

            text[start:end + 1]

        )

    except:

        return None
# ==========================================================
# Validation
# ==========================================================

def validate_gold_sets(response, candidate_resources):
    """
    Validate LLM output for gold set generation.
    """

    if response is None:
        return False

    if not isinstance(response, dict):
        return False

    if "gold_sets" not in response:
        return False

    gold_sets = response["gold_sets"]

    if not isinstance(gold_sets, list):
        return False

    if len(gold_sets) == 0:
        return False

    # Valid resource IDs from candidate pool
    valid_ids = {

        r["resource_id"]

        for r in candidate_resources

    }

    for gold in gold_sets:

        if not isinstance(gold, dict):
            return False

        # ---------------------------------------

        if "resources_used" not in gold:
            return False

        if "answer" not in gold:
            return False

        if "rationale" not in gold:
            return False

        # ---------------------------------------

        if not isinstance(gold["resources_used"], list):
            return False

        if len(gold["resources_used"]) == 0:
            return False

        # ---------------------------------------

        if not isinstance(gold["answer"], str):
            return False

        if len(gold["answer"].strip()) == 0:
            return False

        if not isinstance(gold["rationale"], str):
            return False

        if len(gold["rationale"].strip()) == 0:
            return False

        # ---------------------------------------
        # Validate every selected resource
        # ---------------------------------------

        seen = set()

        for resource in gold["resources_used"]:

            if not isinstance(resource, dict):
                return False

            required = [

                "resource_id",

                "resource_type",

                "summary",

                "description"

            ]

            for key in required:

                if key not in resource:
                    return False

            rid = resource["resource_id"]

            if rid not in valid_ids:
                return False

            if rid in seen:
                return False

            seen.add(rid)

    return True


# ==========================================================
# Prompt Construction
# ==========================================================

def format_resources(resources):
    """
    Convert candidate resources into prompt text.
    """

    formatted = []

    for resource in resources:

        formatted.append(

f"""
Resource ID: {resource["resource_id"]}

Resource Type: {resource["resource_type"]}

Summary:
{resource["summary"]}

Description:
{resource["description"]}
""".strip()

        )

    return "\n\n------------------------------------------------------------\n\n".join(formatted)


def construct_prompt(query_entry):
    """
    Inject query and candidate resources into prompt template.
    """

    prompt = GOLD_PROMPT

    prompt = prompt.replace(

        "{{QUERY}}",

        query_entry["query"]

    )

    prompt = prompt.replace(

        "{{RESOURCES}}",

        format_resources(

            query_entry["candidate_resources"]

        )

    )

    return prompt

# ==========================================================
# Main
# ==========================================================

def main():

    with open(INPUT_JSON, "r", encoding="utf-8") as f:

        queries = json.load(f)

    print("=" * 80)
    print("Gold Set Generation")
    print("=" * 80)
    print()

    final_output = []

    success = 0

    failures = 0

    for query in queries:

        print("=" * 80)
        print(f"Query : {query['query_id']}")
        print(query["query"])
        print("=" * 80)

        prompt = construct_prompt(query)

        gold_sets = None

        for attempt in range(MAX_RETRIES):

            print(
                f"Attempt {attempt+1}/{MAX_RETRIES}"
            )

            try:

                raw_output = generate(prompt)

                parsed = extract_json(raw_output)

                if validate_gold_sets(

                    parsed,

                    query["candidate_resources"]

                ):

                    gold_sets = parsed

                    print("✓ Success")

                    break

                else:

                    print("Validation Failed")

            except Exception as e:

                print(e)

            time.sleep(2)

        if gold_sets is None:

            failures += 1

            print("✗ Failed\n")

            continue

        success += 1

        final_output.append(

            {

                "query_id":

                    query["query_id"],

                "query":

                    query["query"],

                "gold_sets":

                    gold_sets["gold_sets"]

            }

        )

        print()

    OUTPUT_PATH = Path(OUTPUT_JSON)

    OUTPUT_PATH.parent.mkdir(

        parents=True,

        exist_ok=True

    )

    with open(

        OUTPUT_PATH,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            final_output,

            f,

            indent=2,

            ensure_ascii=False

        )

    print()
    print("=" * 80)
    print("Generation Complete")
    print("=" * 80)

    print(f"Successful Queries : {success}")

    print(f"Failed Queries     : {failures}")

    print(f"Total Queries      : {len(queries)}")

    print()

    total_gold_sets = sum(

        len(q["gold_sets"])

        for q in final_output

    )

    print(

        f"Total Gold Sets Generated : "

        f"{total_gold_sets}"

    )

    print()

    print(

        f"Saved to\n{OUTPUT_JSON}"

    )

    print("=" * 80)


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":

    main()
