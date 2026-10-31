import json
import re
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "~/models/Llama-3.1-70B-Instruct"


# Set to True if you have limited GPU memory.
USE_4BIT = False

# ------------------------------------------------------------
# Put the exact query types used by your benchmark here.
# Modify this according to your actual screenshot / taxonomy.
# ------------------------------------------------------------

QUERY_TYPE_DEFINITIONS = """
The benchmark uses exactly these seven query types:

1. Basic Explanation
Purpose:
Requests a direct explanation of a concept.

Example:
"Explain photosynthesis."

A Basic Explanation query primarily asks WHAT something is,
WHAT it means, or for a general conceptual explanation.

--------------------------------------------------

2. Why/How
Purpose:
Seeks causal or procedural understanding.

Examples:
"Why do plants need sunlight?"
"How does photosynthesis occur?"

A Why/How query asks WHY something happens, HOW something works,
or HOW a process occurs.

--------------------------------------------------

3. Misconception
Purpose:
Reflects a common misunderstanding or incorrect belief.

Example:
"Do plants get food from the soil?"

The query must actually reflect, imply, or presuppose a
common misunderstanding or incorrect belief.

For example:
"Do plants get their food from the soil?"
can be a misconception query because it reflects the common
misconception that plants obtain their food directly from soil.

--------------------------------------------------

4. Example-Based
Purpose:
Requests an illustrative example to aid understanding.

Example:
"Explain photosynthesis using a daily-life example."

The query must explicitly request an example, analogy, real-world
example, or illustrative example, or clearly ask for explanation
through an example.

--------------------------------------------------

5. Visual
Purpose:
Requests a diagrammatic or visual explanation.

Example:
"Show me how photosynthesis happens in a leaf."

IMPORTANT:
A query is NOT Visual merely because a visual resource could help
answer it.

The query itself must request a diagram, figure, visual
representation, graph, structure, illustration, or other
explicitly visual explanation.

--------------------------------------------------

6. Revision
Purpose:
Requests a concise summary or recap of a concept.

Example:
"Give me a quick recap of photosynthesis."

The query should primarily ask for a summary, recap, revision,
quick review, or concise overview.

--------------------------------------------------

7. Practice
Purpose:
Requests exercises, assessment questions, or self-testing material.

Example:
"Give me a question to test my understanding of photosynthesis."

The query must request practice questions, exercises, problems,
assessment material, quizzes, or material for testing one's
understanding.

--------------------------------------------------

QUERY CATEGORY EVALUATION:

The input contains a field called query_type.

Determine whether that assigned query_type correctly describes
the query.

Score:

0 = Clearly incorrect category.
The query clearly belongs to another category.

1 = Ambiguous or borderline category.
The assigned category is defensible, but the query could reasonably
belong to another category or the category signal is weak.

2 = Clearly correct category.
The query strongly matches the purpose and characteristics of the
assigned category.

Do not penalize a query simply because another category could also
describe it.

Use the PRIMARY intent of the query.

Examples:

"Explain photosynthesis."
query_type = Basic Explanation
=> 2

"Why do plants need sunlight?"
query_type = Why/How
=> 2

"How does photosynthesis happen?"
query_type = Why/How
=> 2

"Do plants get their food from the soil?"
query_type = Misconception
=> potentially 2, because it reflects a common misconception

"Explain photosynthesis using an example from everyday life."
query_type = Example-Based
=> 2

"Draw a diagram showing photosynthesis in a leaf."
query_type = Visual
=> 2

"Give me a quick revision of photosynthesis."
query_type = Revision
=> 2

"Give me five questions to test my understanding of photosynthesis."
query_type = Practice
=> 2

Do NOT classify a query as Visual simply because the answer could
benefit from a diagram.

Do NOT classify a query as Misconception simply because the answer
could be false or because the query is phrased as a yes/no question.

Do NOT classify a query as Example-Based merely because an example
would make the answer better.

Do NOT classify a query as Practice merely because the query asks
about a problem or application. It must request practice,
assessment, or self-testing material.

Do NOT classify a query as Revision merely because the question
is short. It must ask for a recap, summary, or revision.

Distinguish Basic Explanation from Why/How carefully:
- "What is photosynthesis?" -> Basic Explanation
- "Explain photosynthesis." -> Basic Explanation
- "Why do plants need sunlight?" -> Why/How
- "How does photosynthesis work?" -> Why/How
"""

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading tokenizer...")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)


print("Loading model...")

model_kwargs = {
    "trust_remote_code": True,
    "device_map": "auto",
}

if USE_4BIT:
    model_kwargs.update({
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    })
else:
    model_kwargs["torch_dtype"] = torch.float16


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    **model_kwargs,
    local_files_only=True
)

model.eval()

print("Model loaded.")


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an expert evaluator validating a dataset for an
educational information retrieval benchmark.

You must evaluate each query and its proposed gold resource set.

The benchmark contains educational material from school textbooks.

Your job is NOT to improve, rewrite, or answer the query.

Your job is ONLY to evaluate the quality of the existing benchmark
instance.

You must use ONLY the information provided in the evaluation input
and the query-type definitions.

Do not introduce outside resources as evidence.

{QUERY_TYPE_DEFINITIONS}

SCORING:

For every criterion assign exactly one integer:

0 = clearly poor / incorrect
1 = partially acceptable / ambiguous / mixed
2 = clearly good / correct

The meaning of each criterion is:

1. Query naturalness & educational relevance

Would a real student naturally ask this question?
Is it educationally meaningful and worth asking?

0:
- unnatural, malformed, meaningless, or educationally unhelpful

1:
- understandable but somewhat artificial, vague, or of limited
  educational value

2:
- natural student-like question and educationally useful


2. Correctness of query category

Is the provided query_type actually correct?

0:
- clearly wrong category

1:
- ambiguous or borderline

2:
- clearly correct category

Use the query-type definitions given above.


3. Relevance of selected resources

Does EVERY resource in the proposed gold set genuinely help answer
THIS query?

0:
- most resources are irrelevant, or the set clearly contains
  resources that do not answer the query; one or more resources are
  completely irrelevant.

1:
- most resources are useful but one or more resources have weak
  or indirect relevance

2:
- every resource is relevant to the query

Do not give a 2 merely because a resource mentions the same topic. It is possible
that the resources may discuss the same topic but do not answer the query specifically,
do not give a 2 in that case.


4. Completeness of the explanation

Does the proposed answer fully answer the query? If the query talks about multiple subtopics,
the resources should have sufficient details of each subtopic.

Consider whether any important concepts, relationships, examples,
or steps needed to answer the query are missing.

0:
- major parts of the answer are missing or the answer does not
  answer the question

1:
- partially answers the query but some information is still
  missing

2:
- sufficiently completes answer to the query

For practice category questions i.e. when the user asks to give a question for practice,
the resources should actually have a practice question
or example or activity to form the question. If that is not the case but the resources are still
relevant, then give 1. Only give 2 if the resources actually have a practice question.

5. Redundancy among resources

Determine whether any resource in the selected set is completely overlapping
with another resource for answering the query.

Some overlap between resources is normal and should NOT lower the
score. Resources can discuss the same topic, concept, or facts and
still be non-redundant. Some amount of redundancy can be ignored as long as 
each resource contributes something to the answer.

Ask:

"If this resource were removed, would the remaining resources still
provide exactly the same answer to the query?"

SCORING:

2 = NOT REDUNDANT
Every resource provides useful information for answering the query.
Overlap is acceptable.

1 = PARTIALLY REDUNDANT
There is clear evidence that at least one resource adds very little
information beyond the others, but it is not completely unnecessary.

0 = HIGHLY REDUNDANT
There is clear evidence that at least one resource is unnecessary:
the other resources already provide essentially enough information
to answer the query.

IMPORTANT:

Do NOT assign 1 or 0 merely because:
- resources discuss the same topic;
- resources share concepts;
- resources contain some repeated facts;
- resources have similar summaries.

Only assign 1 or 0 when you can identify resources whose removal
would have no effect on the answer.

If you cannot identify such a resource from the summaries, assign 2.

It is possible that two resources revolve around the same topic but
one of them gives more details in which case they are not redundant.

The summaries may omit information contained in the actual
resources. Therefore, similar summaries are NOT sufficient evidence
of redundancy.

6. Consistency with grade level

Is the query and selected material appropriate for the specified
grade level?

0:
- clearly inappropriate for the grade

1:
- somewhat mismatched, unusually advanced/simple, or borderline

2:
- appropriate for the stated grade level


IMPORTANT EVALUATION PRINCIPLES:

- Evaluate the actual proposed resources, not resources you wish
  had been selected.
- A resource should not receive a high relevance score merely because
  it contains a keyword matching the query.
- Distinguish topical similarity from useful information.
- For completeness, evaluate whether the provided answer addresses
  the actual query.
- For resource redundancy, judge whether removing a resource would
  meaningfully reduce the information available.
- Be conservative. Use 1 when the evidence is genuinely ambiguous.
- Do not automatically give high scores to all examples.
- Do not assume the generated answer is correct simply because it
  sounds plausible.


OUTPUT FORMAT:

Return ONLY valid JSON.

Do not include markdown.
Do not include ```json.
Do not include any text before or after the JSON.

The JSON must have exactly this structure:

{{
  "scores": {{
    "query_naturalness_educational_relevance": 0,
    "query_category_correctness": 0,
    "selected_resource_relevance": 0,
    "explanation_completeness": 0,
    "resource_redundancy": 0,
    "grade_level_consistency": 0
  }},
  "reasons": {{
    "query_naturalness_educational_relevance": "...",
    "query_category_correctness": "...",
    "selected_resource_relevance": "...",
    "explanation_completeness": "...",
    "resource_redundancy": "...",
    "grade_level_consistency": "..."
  }},
  "overall_comment": "..."
}}

Every score MUST be exactly 0, 1, or 2.
"""


# ============================================================
# BUILD INPUT
# ============================================================

def build_user_prompt(item, gold_set, gold_set_index):

    query_id = item.get("query_id", "")
    query = item.get("query", "")
    query_type = item.get("query_type", "")
    bundle_type = item.get("bundle_type", "")
    variant_style = item.get("variant_style", "")
    grade = item.get("grade", "")

    resources = gold_set.get("resources_used", [])

    resource_text = []

    for i, resource in enumerate(resources):

        resource_text.append(
            f"""
RESOURCE {i + 1}

resource_id:
{resource.get("resource_id", "")}

resource_type:
{resource.get("resource_type", "")}

document_id:
{resource.get("document_id", "")}

summary:
{resource.get("summary", "")}

description:
{resource.get("description", "")}

concepts:
{", ".join(resource.get("concepts", []))}
"""
        )

    resources_block = "\n".join(resource_text)

    answer = gold_set.get("answer", "")
    rationale = gold_set.get("rationale", "")

    prompt = f"""
Evaluate the following benchmark instance.

QUERY ID:
{query_id}

GOLD SET INDEX:
{gold_set_index}

QUERY:
{query}

QUERY TYPE:
{query_type}

BUNDLE TYPE:
{bundle_type}

VARIANT STYLE:
{variant_style}

GRADE:
{grade}

--------------------------------------------------
SELECTED RESOURCES
--------------------------------------------------

{resources_block}

--------------------------------------------------
PROPOSED ANSWER / GOLD EXPLANATION
--------------------------------------------------

{answer}

--------------------------------------------------
RATIONALE FOR RESOURCE SELECTION
--------------------------------------------------

{rationale}

Now evaluate all six criteria.

Remember:
- score every criterion from 0, 1, or 2
- provide a short reason for every score
- return ONLY the required JSON
"""

    return prompt


# ============================================================
# GENERATE MODEL RESPONSE
# ============================================================

def generate(prompt, max_new_tokens=1000):

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

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    # Move tensors to the model's first device.
    if hasattr(model, "device"):
        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items()
        }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()


# ============================================================
# EXTRACT JSON
# ============================================================

def extract_json(text):

    text = text.strip()

    # First attempt: entire response is JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove markdown fences if model added them.
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*$",
        "",
        text
    )

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find the first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# VALIDATE OUTPUT
# ============================================================

REQUIRED_CRITERIA = [
    "query_naturalness_educational_relevance",
    "query_category_correctness",
    "selected_resource_relevance",
    "explanation_completeness",
    "resource_redundancy",
    "grade_level_consistency",
]


def validate_model_output(result):

    if result is None:
        return False, "Could not parse JSON"

    if "scores" not in result:
        return False, "Missing scores"

    if "reasons" not in result:
        return False, "Missing reasons"

    scores = result["scores"]

    for criterion in REQUIRED_CRITERIA:

        if criterion not in scores:
            return False, f"Missing score: {criterion}"

        score = scores[criterion]

        if not isinstance(score, int):
            return False, f"Non-integer score: {criterion}"

        if score not in [0, 1, 2]:
            return False, f"Invalid score for {criterion}: {score}"

    return True, ""


# ============================================================
# MAIN VALIDATION LOOP
# ============================================================

def validate_dataset(input_file, output_file):

    print(f"Loading dataset: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} queries.")

    # ------------------------------------------------------------
    # LOAD EXISTING RESULTS IF THEY EXIST
    # ------------------------------------------------------------

    output_path = Path(output_file)

    if output_path.exists():
        print(f"Existing output found: {output_file}")

        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)

            if not isinstance(results, list):
                print("WARNING: Existing output is not a list. Starting fresh.")
                results = []

        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: Could not read existing output: {e}")
            print("Starting fresh.")
            results = []

    else:
        print("No existing output found. Starting fresh.")
        results = []

    # ------------------------------------------------------------
    # BUILD INDEX OF ALREADY COMPLETED GOLD SETS
    # ------------------------------------------------------------

    completed = set()

    for query_result in results:
        query_id = query_result.get("query_id", "")

        for gold_result in query_result.get("gold_sets", []):
            gold_set_index = gold_result.get("gold_set_index")

            if gold_set_index is not None:
                completed.add((query_id, gold_set_index))

    print(f"Already completed gold sets: {len(completed)}")

    # ------------------------------------------------------------
    # INDEX EXISTING QUERY RESULTS
    # This lets us append new gold sets to an existing query.
    # ------------------------------------------------------------

    query_result_map = {
        result.get("query_id"): result
        for result in results
    }

    total = 0
    skipped = 0
    newly_evaluated = 0
    failed = 0

    # ------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------

    for item_index, item in enumerate(dataset):

        query_id = item.get("query_id", "")
        query = item.get("query", "")
        gold_sets = item.get("gold_sets", [])

        print()
        print("=" * 80)
        print(
            f"[{item_index + 1}/{len(dataset)}] "
            f"{query_id}: {query}"
        )
        print(f"Number of gold sets: {len(gold_sets)}")
        print("=" * 80)

        # --------------------------------------------------------
        # GET OR CREATE QUERY RESULT
        # --------------------------------------------------------

        if query_id in query_result_map:

            query_result = query_result_map[query_id]

            # Update metadata in case input changed
            query_result["query"] = query
            query_result["query_type"] = item.get("query_type")
            query_result["grade"] = item.get("grade")

        else:

            query_result = {
                "query_id": query_id,
                "query": query,
                "query_type": item.get("query_type"),
                "grade": item.get("grade"),
                "gold_sets": []
            }

            results.append(query_result)
            query_result_map[query_id] = query_result

        # --------------------------------------------------------
        # EVALUATE GOLD SETS
        # --------------------------------------------------------

        for gold_set_index, gold_set in enumerate(gold_sets):

            total += 1

            # ----------------------------------------------------
            # SKIP ALREADY COMPLETED
            # ----------------------------------------------------

            if (query_id, gold_set_index) in completed:

                skipped += 1

                print(
                    f"  Skipping gold set "
                    f"{gold_set_index + 1}/{len(gold_sets)} "
                    f"(already evaluated)"
                )

                continue

            print(
                f"  Evaluating gold set "
                f"{gold_set_index + 1}/{len(gold_sets)}..."
            )

            # ----------------------------------------------------
            # GENERATE MODEL RESPONSE
            # ----------------------------------------------------

            prompt = build_user_prompt(
                item,
                gold_set,
                gold_set_index
            )

            raw_response = generate(prompt)

            parsed = extract_json(raw_response)

            valid, error = validate_model_output(parsed)

            if not valid:

                failed += 1

                print(
                    f"  WARNING: invalid model output: {error}"
                )

                gold_result = {
                    "gold_set_index": gold_set_index,
                    "validation_status": "FAILED",
                    "error": error,
                    "raw_model_output": raw_response
                }

            else:

                print("  Validation successful.")

                print(
                    "  Scores:",
                    parsed["scores"]
                )

                gold_result = {
                    "gold_set_index": gold_set_index,
                    "validation_status": "SUCCESS",
                    "scores": parsed["scores"],
                    "reasons": parsed["reasons"],
                    "overall_comment": parsed.get(
                        "overall_comment",
                        ""
                    ),
                    "raw_model_output": raw_response
                }

            # ----------------------------------------------------
            # APPEND NEW RESULT
            # ----------------------------------------------------

            query_result["gold_sets"].append(gold_result)

            # Mark as completed so it cannot be processed again
            completed.add((query_id, gold_set_index))

            newly_evaluated += 1

            # ----------------------------------------------------
            # SAVE IMMEDIATELY
            # ----------------------------------------------------

            with open(
                output_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    results,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            print(
                f"  Saved progress to {output_file}"
            )

    # ------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    print(f"Queries in input: {len(dataset)}")
    print(f"Gold sets encountered: {total}")
    print(f"Gold sets skipped: {skipped}")
    print(f"New gold sets evaluated: {newly_evaluated}")
    print(f"Failed model outputs: {failed}")

    print(f"Results saved to: {output_file}")


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        help="Input gold-set JSON file"
    )

    parser.add_argument(
        "output",
        help="Output validation JSON file"
    )

    args = parser.parse_args()

    validate_dataset(
        args.input,
        args.output
    )


if __name__ == "__main__":
    main()
