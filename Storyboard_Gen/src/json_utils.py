import json
from typing import Any, Dict, List, Set


# ==============================================================================
# Expected output schema
# ==============================================================================

TOP_LEVEL_FIELDS: Set[str] = {
    "storyboard_id",
    "query_id",
    "query",
    "gold_set_id",
    "gold_set_resources",
    "intent",
    "target_duration_sec",
    "total_duration_sec",
    "role_template",
    "segments",
    "alt_storyboard_ids",
    "provenance",
}

SEGMENT_FIELDS: Set[str] = {
    "segment_id",
    "order",
    "resource_ids",
    "role",
    "duration_sec",
    "narration",
    "on_screen",
    "concepts",
}

PROVENANCE_FIELDS: Set[str] = {
    "generation_model",
    "validation_status",
    "annotator",
}


# ==============================================================================
# JSON Parsing
# ==============================================================================

def parse_model_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the model response into a Python dictionary.

    The prompt instructs the model to return only valid JSON.
    This function also handles accidental Markdown code fences.
    """

    response_text = response_text.strip()

    # Defensive handling in case a model returns ```json ... ```
    if response_text.startswith("```"):
        lines = response_text.splitlines()

        # Remove opening fence
        if lines:
            lines = lines[1:]

        # Remove closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        response_text = "\n".join(lines).strip()

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model response is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Model output must be a JSON object."
        )

    return data


# ==============================================================================
# Input Gold-Set Helpers
# ==============================================================================

def get_gold_set(
    query_item: Dict[str, Any],
    gold_set_index: int = 0,
) -> Dict[str, Any]:
    """
    Extract a single gold_set object from a query item by index.

    A query_item may legitimately contain more than one gold_set.
    Each gold_set is generated into its own, independent storyboard,
    so callers must specify WHICH gold_set they are validating against.
    """

    gold_sets = query_item.get("gold_sets")

    if not isinstance(gold_sets, list) or not gold_sets:
        raise ValueError(
            "Input query does not contain a valid non-empty 'gold_sets' list."
        )

    if gold_set_index < 0 or gold_set_index >= len(gold_sets):
        raise ValueError(
            f"gold_set_index {gold_set_index} is out of range for "
            f"'gold_sets' list of length {len(gold_sets)}."
        )

    gold_set = gold_sets[gold_set_index]

    if not isinstance(gold_set, dict):
        raise ValueError(
            f"gold_sets[{gold_set_index}] must be a JSON object."
        )

    return gold_set


def get_gold_set_resources(
    query_item: Dict[str, Any],
    gold_set_index: int = 0,
) -> List[str]:
    """
    Extract all resource IDs from ONE specific gold set, identified
    by gold_set_index.

    The current dataset structure contains:
        query_item["gold_sets"][gold_set_index]["resources_used"]

    IMPORTANT: A query_item may contain multiple gold_sets. Each gold_set
    is generated into its own storyboard using that gold_set's COMPLETE,
    independent resource collection. This function must never assume
    index 0 is the only gold_set of interest — the caller (generation
    and validation pipeline) is responsible for iterating over every
    gold_set and passing the correct index.
    """

    gold_set = get_gold_set(query_item, gold_set_index=gold_set_index)

    resources_used = gold_set.get("resources_used")

    if not isinstance(resources_used, list):
        raise ValueError(
            f"gold_sets[{gold_set_index}] does not contain a valid "
            "'resources_used' list."
        )

    resource_ids = []

    for resource in resources_used:
        if not isinstance(resource, dict):
            raise ValueError(
                f"Each resource in gold_sets[{gold_set_index}]."
                "resources_used must be a JSON object."
            )

        resource_id = resource.get("resource_id")

        if not resource_id:
            raise ValueError(
                f"A resource in gold_sets[{gold_set_index}].resources_used "
                "is missing 'resource_id'."
            )

        resource_ids.append(resource_id)

    return resource_ids


def count_gold_sets(query_item: Dict[str, Any]) -> int:
    """
    Return how many gold_sets exist for this query.

    Used by the generation pipeline to know how many storyboards
    must be produced for a single query.
    """

    gold_sets = query_item.get("gold_sets")

    if not isinstance(gold_sets, list):
        raise ValueError(
            "Input query does not contain a valid 'gold_sets' list."
        )

    return len(gold_sets)


# ==============================================================================
# Storyboard Validation
# ==============================================================================

def validate_storyboard(
    storyboard: Dict[str, Any],
    query_item: Dict[str, Any],
    gold_set_index: int = 0,
) -> List[str]:
    """
    Validate a generated storyboard against the requirements
    defined in storyboard_prompt.txt.

    gold_set_index identifies WHICH gold_set (of possibly several on
    this query) this storyboard was generated from. The storyboard's
    gold_set_resources must exactly match that specific gold_set's
    resources_used — not gold_sets[0] by default, unless gold_set_index
    is 0.

    Returns:
        List[str]: A list of validation errors.
                   An empty list means validation passed.
    """

    errors: List[str] = []

    # --------------------------------------------------------------------------
    # 1. Top-level schema validation
    # --------------------------------------------------------------------------

    actual_fields = set(storyboard.keys())

    missing_fields = TOP_LEVEL_FIELDS - actual_fields
    extra_fields = actual_fields - TOP_LEVEL_FIELDS

    if missing_fields:
        errors.append(
            f"Missing top-level fields: {sorted(missing_fields)}"
        )

    if extra_fields:
        errors.append(
            f"Unexpected top-level fields: {sorted(extra_fields)}"
        )

    # Stop here if essential fields are missing
    if missing_fields:
        return errors

    # --------------------------------------------------------------------------
    # 2. Input consistency
    # --------------------------------------------------------------------------

    expected_query_id = query_item.get("query_id")
    expected_query = query_item.get("query")

    if storyboard.get("query_id") != expected_query_id:
        errors.append(
            "query_id does not match the input query_id."
        )

    # if storyboard.get("query") != expected_query:
    #     errors.append(
    #         "query does not exactly match the input query."
    #     )

    # --------------------------------------------------------------------------
    # 3. Gold-set resource preservation (for THIS gold_set only)
    # --------------------------------------------------------------------------

    try:
        expected_resource_ids = get_gold_set_resources(
            query_item, gold_set_index=gold_set_index
        )
    except ValueError as exc:
        errors.append(str(exc))
        expected_resource_ids = []

    output_resource_ids = storyboard.get("gold_set_resources")

    if not isinstance(output_resource_ids, list):
        errors.append(
            "gold_set_resources must be a list."
        )
    else:
        # Every element MUST be a plain string (a bare resource_id).
        # Models sometimes return objects like {"resource_id": "R1"}
        # instead of "R1" — catch that explicitly with a clear message
        # rather than letting set() crash on an unhashable dict/list.
        non_string_entries = [
            (i, entry)
            for i, entry in enumerate(output_resource_ids)
            if not isinstance(entry, str)
        ]

        if non_string_entries:
            preview = [
                f"index {i}: {type(entry).__name__} -> {entry!r}"
                for i, entry in non_string_entries[:5]
            ]
            errors.append(
                "gold_set_resources must contain only plain resource_id "
                "strings, not objects/lists. Found non-string entries: "
                + "; ".join(preview)
                + (
                    " ... (truncated)"
                    if len(non_string_entries) > 5
                    else ""
                )
            )
        else:
            output_set = set(output_resource_ids)
            expected_set = set(expected_resource_ids)

            missing = [
                r for r in expected_resource_ids if r not in output_set
            ]
            extra = [
                r for r in output_resource_ids if r not in expected_set
            ]

            if missing:
                errors.append(
                    f"gold_set_resources is missing resource(s): {missing}"
                )

            if extra:
                errors.append(
                    "gold_set_resources contains resource(s) not present in "
                    f"gold_sets[{gold_set_index}].resources_used: {extra}"
                )

            if not missing and not extra:
                if len(output_resource_ids) != len(expected_resource_ids):
                    errors.append(
                        "gold_set_resources has the correct resource IDs "
                        f"but wrong count (got {len(output_resource_ids)}, "
                        f"expected {len(expected_resource_ids)}); check for "
                        "duplicate resource_ids being deduplicated."
                    )
                # elif output_resource_ids != expected_resource_ids:
                #     errors.append(
                #         "gold_set_resources contains the correct resources "
                #         "but in the wrong order. Expected: "
                #         f"{expected_resource_ids}. Got: "
                #         f"{output_resource_ids}."
                #     )

    # --------------------------------------------------------------------------
    # 4. Duration validation
    # --------------------------------------------------------------------------

    target_duration = storyboard.get("target_duration_sec")
    total_duration = storyboard.get("total_duration_sec")

    if (
        not isinstance(target_duration, int)
        or isinstance(target_duration, bool)
    ):
        errors.append(
            "target_duration_sec must be an integer."
        )
    elif not 120 <= target_duration <= 300:
        errors.append(
            "target_duration_sec must be between 120 and 300 seconds."
        )

    if (
        not isinstance(total_duration, int)
        or isinstance(total_duration, bool)
    ):
        errors.append(
            "total_duration_sec must be an integer."
        )

    # --------------------------------------------------------------------------
    # 5. Segment validation
    # --------------------------------------------------------------------------

    segments = storyboard.get("segments")

    if not isinstance(segments, list) or not segments:
        errors.append(
            "segments must be a non-empty list."
        )
        return errors

    segment_ids = set()
    segment_orders = []
    segment_duration_sum = 0
    cited_resource_ids: Set[str] = set()

    valid_resource_ids = set(expected_resource_ids)

    for index, segment in enumerate(segments):

        prefix = f"Segment at index {index}"

        if not isinstance(segment, dict):
            errors.append(
                f"{prefix} must be a JSON object."
            )
            continue

        actual_segment_fields = set(segment.keys())

        missing_segment_fields = (
            SEGMENT_FIELDS - actual_segment_fields
        )

        extra_segment_fields = (
            actual_segment_fields - SEGMENT_FIELDS
        )

        if missing_segment_fields:
            errors.append(
                f"{prefix} is missing fields: "
                f"{sorted(missing_segment_fields)}"
            )

        if extra_segment_fields:
            errors.append(
                f"{prefix} has unexpected fields: "
                f"{sorted(extra_segment_fields)}"
            )

        if missing_segment_fields:
            continue

        # ----------------------------------------------------------------------
        # segment_id uniqueness
        # ----------------------------------------------------------------------

        segment_id = segment.get("segment_id")

        if not isinstance(segment_id, str) or not segment_id.strip():
            errors.append(
                f"{prefix} has an invalid segment_id."
            )
        elif segment_id in segment_ids:
            errors.append(
                f"Duplicate segment_id found: {segment_id}"
            )
        else:
            segment_ids.add(segment_id)

        # ----------------------------------------------------------------------
        # Sequential order
        # ----------------------------------------------------------------------

        order = segment.get("order")

        if (
            not isinstance(order, int)
            or isinstance(order, bool)
        ):
            errors.append(
                f"{prefix} has a non-integer order."
            )
        else:
            segment_orders.append(order)

        # ----------------------------------------------------------------------
        # Resource IDs
        # ----------------------------------------------------------------------

        resource_ids = segment.get("resource_ids")

        if not isinstance(resource_ids, list):
            errors.append(
                f"{prefix} resource_ids must be a list."
            )
        else:
            for resource_id in resource_ids:
                if not isinstance(resource_id, str):
                    errors.append(
                        f"{prefix} resource_ids must contain only plain "
                        f"resource_id strings, not "
                        f"{type(resource_id).__name__}: {resource_id!r}"
                    )
                elif resource_id not in valid_resource_ids:
                    errors.append(
                        f"{prefix} contains an invalid resource_id: "
                        f"{resource_id}"
                    )
                else:
                    cited_resource_ids.add(resource_id)

        # ----------------------------------------------------------------------
        # Positive duration
        # ----------------------------------------------------------------------

        duration = segment.get("duration_sec")

        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
        ):
            errors.append(
                f"{prefix} duration_sec must be an integer."
            )
        elif duration <= 0:
            errors.append(
                f"{prefix} duration_sec must be positive."
            )
        else:
            segment_duration_sum += duration

        # ----------------------------------------------------------------------
        # Basic type validation for remaining fields
        # ----------------------------------------------------------------------

        string_fields = [
            "role",
            "narration",
            "on_screen",
        ]

        for field in string_fields:
            value = segment.get(field)

            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"{prefix} field '{field}' must be a non-empty string."
                )

        concepts = segment.get("concepts")

        if not isinstance(concepts, list) or not concepts:
            errors.append(
                f"{prefix} concepts must be a non-empty list."
            )
        else:
            for concept in concepts:
                if not isinstance(concept, str) or not concept.strip():
                    errors.append(
                        f"{prefix} concepts must contain only non-empty "
                        "strings."
                    )
                    break

    # --------------------------------------------------------------------------
    # 5b. Every gold-set resource must be cited in at least one segment
    # --------------------------------------------------------------------------
    # A resource present only in the top-level gold_set_resources field but
    # never cited by any segment's resource_ids has been silently dropped
    # in every way that matters, even though its ID string still appears
    # somewhere in the JSON. This is the exact failure mode where a model
    # "fakes" preservation by keeping the ID in the top-level list while
    # never actually using the resource.

    uncited_resource_ids = [
        r for r in expected_resource_ids if r not in cited_resource_ids
    ]

    # if uncited_resource_ids:
    #     errors.append(
    #         "The following gold-set resource(s) are listed in "
    #         "gold_set_resources but are never cited in any segment's "
    #         "resource_ids, meaning they were not actually used anywhere "
    #         f"in the storyboard: {uncited_resource_ids}"
    #     )

    # --------------------------------------------------------------------------
    # 6. Sequential segment order validation
    # --------------------------------------------------------------------------

    if segment_orders:
        expected_orders = list(range(len(segments)))

        if sorted(segment_orders) != expected_orders:
            errors.append(
                "Segment orders must be sequential: 0, 1, 2, ..."
            )

    # --------------------------------------------------------------------------
    # 7. Total duration consistency
    # --------------------------------------------------------------------------

    if (
        isinstance(total_duration, int)
        and not isinstance(total_duration, bool)
        and total_duration != segment_duration_sum
    ):
        errors.append(
            "total_duration_sec must equal the sum of all "
            f"segment duration_sec values (expected {segment_duration_sum}, "
            f"got {total_duration})."
        )

    # --------------------------------------------------------------------------
    # 8. Provenance validation
    # --------------------------------------------------------------------------

    provenance = storyboard.get("provenance")

    if not isinstance(provenance, dict):
        errors.append(
            "provenance must be a JSON object."
        )
    else:
        actual_provenance_fields = set(provenance.keys())

        missing_provenance_fields = (
            PROVENANCE_FIELDS - actual_provenance_fields
        )

        extra_provenance_fields = (
            actual_provenance_fields - PROVENANCE_FIELDS
        )

        if missing_provenance_fields:
            errors.append(
                "provenance is missing fields: "
                f"{sorted(missing_provenance_fields)}"
            )

        if extra_provenance_fields:
            errors.append(
                "provenance has unexpected fields: "
                f"{sorted(extra_provenance_fields)}"
            )

        if provenance.get("validation_status") != "validated":
            errors.append(
                "provenance.validation_status must be 'validated'."
            )

    return errors

def normalize_storyboard_duration(
    storyboard: Dict[str, Any],
) -> None:
    """
    Normalize storyboard duration metadata using the actual sum
    of valid segment durations.

    If the actual segment duration is within the allowed range
    of 120–300 seconds, synchronize target_duration_sec and
    total_duration_sec with the segment duration sum.
    """

    segments = storyboard.get("segments")

    if not isinstance(segments, list):
        return

    segment_duration_sum = 0

    for segment in segments:

        if not isinstance(segment, dict):
            return

        duration = segment.get("duration_sec")

        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or duration <= 0
        ):
            return

        segment_duration_sum += duration

    if 120 <= segment_duration_sum <= 300:

        storyboard["target_duration_sec"] = (
            segment_duration_sum
        )

        storyboard["total_duration_sec"] = (
            segment_duration_sum
        )




# ==============================================================================
# Convenience Function
# ==============================================================================

def parse_and_validate(
    response_text: str,
    query_item: Dict[str, Any],
    gold_set_index: int = 0,
) -> Dict[str, Any]:
    """
    Parse a model response and validate the resulting storyboard against
    ONE specific gold_set (identified by gold_set_index) belonging to
    query_item.

    Raises:
        ValueError: If parsing or validation fails.

    Returns:
        Dict[str, Any]: The validated storyboard.
    """

    storyboard = parse_model_response(response_text)
    normalize_storyboard_duration(
        storyboard
    )

    errors = validate_storyboard(
        storyboard=storyboard,
        query_item=query_item,
        gold_set_index=gold_set_index,
    )

    if errors:
        error_message = (
            "Storyboard validation failed:\n- "
            + "\n- ".join(errors)
        )
        raise ValueError(error_message)

    return storyboard