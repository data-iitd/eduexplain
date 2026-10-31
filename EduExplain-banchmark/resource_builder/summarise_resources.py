
import json
import os
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ==========================================================
# CONFIG
# ==========================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SCRIPT_DIR=Path(__file__).resolve().parent
INPUT_ROOT = SCRIPT_DIR/"data_resources"    """the path to the resources after parsing"""

OUTPUT_ROOT =SCRIPT_DIR/"data_resources_summmary"
SUMMARY_RESOURCE_TYPES = {
    "explanation",
    "example",
    "activity",
}

MAX_INPUT_TOKENS = 6000
MAX_NEW_TOKENS = 120

# ==========================================================
# PROMPTS
# ==========================================================

def build_prompt(resource):

    resource_type = resource["resource_type"]

    if resource_type == "example":

        return f"""
You are creating educational metadata for an NCERT benchmark.

Summarize this worked example.

Requirements:
- 2-4 sentences.
- Mention the concept being demonstrated.
- Mention the type of problem being solved.
- Preserve scientific accuracy.
- Do not reproduce calculations.
- Do not add information not present in the resource.

Resource:

{resource["text"]}
"""

    return f"""
You are creating educational metadata for an NCERT benchmark.

Create a concise educational summary of the concepts taught in this resource.

Requirements:
- Focus on scientific ideas and principles.
- Preserve scientific accuracy.
- Avoid phrases such as:
  "This chapter discusses..."
  "This resource explains..."
  "The text describes..."
- Write directly as educational content.
- Use 2-3 sentences.
- For longer resources use at most 5 sentences.
- Do not add information not present in the resource.

Resource:

{resource["text"]}
"""

# ==========================================================
# MODEL
# ==========================================================

def load_model():

    print("Loading model...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    return model, tokenizer

# ==========================================================
# SUMMARY GENERATION
# ==========================================================

def generate_summary(
    model,
    tokenizer,
    prompt,
):

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.2,
    )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

# ==========================================================
# FILE PROCESSING
# ==========================================================

def load_resources(
    input_file,
    output_file,
):

    if output_file.exists():

        with open(
            output_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_resources(
    resources,
    output_file,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            resources,
            f,
            indent=2,
            ensure_ascii=False,
        )

# ==========================================================
# MAIN
# ==========================================================

def main():

    model, tokenizer = load_model()

    all_files = sorted(
        INPUT_ROOT.rglob("*.json")
    )

    total_files = len(all_files)

    print()
    print(f"Found {total_files} files")
    print()

    for file_idx, input_file in enumerate(
        all_files,
        start=1
    ):

        relative_path = input_file.relative_to(
            INPUT_ROOT
        )

        output_file = (
            OUTPUT_ROOT / relative_path
        )

        resources = load_resources(
            input_file,
            output_file,
        )

        print()
        print("=" * 80)
        print(
            f"[{file_idx}/{total_files}] "
            f"{relative_path}"
        )
        print(
            f"Resources: {len(resources)}"
        )

        for resource_idx, resource in enumerate(
            resources,
            start=1
        ):

            if resource.get("summary"):
                continue

            if (
                resource["resource_type"]
                not in SUMMARY_RESOURCE_TYPES
            ):
                continue

            print(
                f"  [{resource_idx}/{len(resources)}] "
                f"{resource['resource_id']}"
            )

            try:

                summary = generate_summary(
                    model,
                    tokenizer,
                    build_prompt(resource),
                )

                resource["summary"] = summary

                save_resources(
                    resources,
                    output_file,
                )

            except Exception as e:

                print(
                    f"ERROR: {resource['resource_id']}"
                )
                print(e)

                save_resources(
                    resources,
                    output_file,
                )

        save_resources(
            resources,
            output_file,
        )

    print()
    print("=" * 80)
    print("ALL FILES COMPLETED")


if __name__ == "__main__":
    main()
