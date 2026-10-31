# 05 - Candidate Pool and Gold Explanation Set Generation

## Purpose

The objective of this stage is to transform a learner query into one or more **gold explanation sets**.

This process consists of two sequential steps.

1. **Candidate Pool Generation**
   - Retrieve a manageable set of potentially relevant educational resources.

2. **Gold Explanation Generation**
   - Construct one or more explanation sets from the candidate pool.

The candidate pool reduces the search space, while the Large Language Model performs semantic reasoning to identify combinations of resources that together answer the learner query.

---

# Position in the Pipeline

```
Learner Query
      │
      ▼
Candidate Generation
      │
      ▼
Candidate Resource Pool
      │
      ▼
Gold Set Generation
      │
      ▼
Gold Explanation Sets
```

---

# Stage 1 — Candidate Pool Generation

## Objective

Construct a high-recall candidate pool that contains the majority of resources required to answer a learner query.

Rather than searching the entire resource corpus during explanation generation, the LLM operates on this significantly smaller candidate pool.

---

# Retrieval Strategy

The candidate pool is generated using two complementary retrieval methods.

```
                  Query
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
 Concept Retrieval       Dense Retrieval
        │                       │
        └───────────┬───────────┘
                    ▼
              Merge Results
                    │
                    ▼
             Remove Duplicates
                    │
                    ▼
             Candidate Pool
```

---

## Concept Retrieval

Concept retrieval exploits the concept annotations associated with every educational resource.

Resources sharing one or more concepts with the learner query are retrieved directly.

This provides high precision and ensures strong curriculum alignment.

---

## Dense Retrieval

Dense retrieval captures semantic similarity beyond exact concept matches.

Each learner query is embedded into the same semantic space as the educational resources.

The nearest neighbouring resources are retrieved based on embedding similarity.

Dense retrieval complements concept retrieval by identifying semantically related resources even when explicit concept overlap is absent.

---

# Why Not BM25?

Several retrieval combinations were experimentally evaluated during benchmark construction.

Examples included

- Concept + Dense
- Concept + Dense + BM25
- Dense + BM25

BM25 consistently introduced lexical noise by retrieving resources with similar vocabulary but weaker semantic relevance.

Consequently, BM25 was excluded from the benchmark construction pipeline.

The final candidate generation strategy combines

- Concept Retrieval
- Dense Retrieval

only.

---

# Candidate Pool Construction

The retrieved resources from both retrieval methods are merged.

Duplicate resources are removed using their unique resource identifiers.

The resulting candidate pool is then truncated to a maximum size.

Current candidate pool size

```
30 resources
```

This provides sufficient diversity while maintaining a manageable search space for the LLM.

---

# Candidate Pool Schema

Each candidate resource contains the complete metadata required for explanation generation.

Typical fields include

- resource_id
- resource_type
- summary
- description
- chapter metadata
- concept annotations

Providing complete resource information enables the LLM to reason over educational content without requiring additional corpus lookups.

---

# Output

```
Learner Query

↓

Candidate Pool

↓

30 Educational Resources
```

This candidate pool serves as the sole input to the explanation generation stage.

---

# Stage 2 — Gold Explanation Set Generation

## Objective

Generate one or more educational explanation sets capable of answering the learner query.

Unlike traditional retrieval benchmarks, EduExplain evaluates retrieval of **resource sets** rather than individual documents.

A single learner query may therefore have multiple valid explanation sets.

---

# Input

The explanation generator receives

- Learner Query
- Candidate Resource Pool

No additional retrieval is performed at this stage.

---

# Explanation Generation Workflow

```
Learner Query

+

Candidate Pool

↓

Prompt Construction

↓

Large Language Model

↓

Gold Explanation Sets

↓

Validation

↓

Final Output
```

---

# Gold Explanation Set

Each generated explanation set consists of

```
Selected Resources

+

Educational Explanation

+

Selection Rationale
```

---

## Selected Resources

The model selects one or more resources from the candidate pool.

Each selected resource contains

- resource_id
- resource_type
- summary
- description

The complete resource metadata is retained to simplify manual inspection during benchmark construction.

---

## Educational Explanation

The explanation should

- answer the learner query,
- integrate information from all selected resources,
- remain pedagogically appropriate,
- avoid unsupported information.

---

## Selection Rationale

Alongside every explanation, the model generates a rationale describing

- why each resource was selected,
- how the selected resources complement one another,
- why together they sufficiently answer the learner query.

The rationale assists future human validation and provides additional interpretability during benchmark construction.

---

# Multiple Gold Sets

A learner query may have multiple equally valid explanation sets.

For example

```
Query

↓

Gold Set A

Gold Set B

Gold Set C
```

The model is instructed to generate every meaningfully distinct explanation set rather than a single answer.

No fixed upper limit is imposed on the number of generated explanation sets.

---

# Validation

Generated outputs undergo structural validation before being accepted.

Validation verifies

- valid JSON format,
- required fields,
- valid resource identifiers,
- non-empty explanations,
- non-empty rationales,
- valid explanation set structure.

Outputs failing validation are regenerated until either a valid explanation is produced or the retry limit is exceeded.

---

# Output Schema

Conceptually, every query produces

```json
{
    "query_id": "...",

    "gold_sets": [

        {

            "resources_used": [

                {
                    "resource_id": "...",
                    "resource_type": "...",
                    "summary": "...",
                    "description": "..."
                }

            ],

            "answer": "...",

            "rationale": "..."
        }

    ]
}
```

---

# Design Principles

## High Recall Candidate Pool

The retrieval stage prioritizes ensuring that relevant resources are present within the candidate pool.

---

## Resource-Centric Retrieval

Explanation sets are constructed from educational resources rather than arbitrary text chunks.

---

## Multiple Valid Solutions

The benchmark acknowledges that educational questions often admit multiple equally valid explanations.

---

## Explainability

Every explanation is accompanied by an explicit rationale describing resource selection.

---

## Human Interpretability

Selected resources retain summaries and descriptions to facilitate efficient manual inspection during benchmark construction.

---

# Responsibilities Summary

| Stage | Responsibility |
|---------|---------------|
| Candidate Generation | Retrieve relevant educational resources |
| Merge | Combine retrieval outputs |
| Deduplication | Remove duplicate resources |
| Candidate Pool | Limit search space |
| Gold Generation | Construct explanation sets |
| Validation | Ensure structural correctness |

---

# Key Takeaway

This stage transforms learner queries into validated educational explanation sets through a two-stage process.

First, a high-recall candidate pool is constructed using concept-based and dense semantic retrieval. Second, a Large Language Model reasons over this candidate pool to generate one or more explanation sets, each consisting of complementary educational resources, a synthesized explanation, and a rationale describing the resource selection process.

These validated explanation sets form the core annotations of the EduExplain benchmark.
