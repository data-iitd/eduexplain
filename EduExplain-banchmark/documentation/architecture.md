# EduExplain System Architecture

## Overview

EduExplain is designed as a modular pipeline for constructing a benchmark for **Multimodal Explanation Set Retrieval (ESR)** from educational resources.

The pipeline is intentionally decomposed into independent stages. Each stage communicates with the next using standardized intermediate representations, allowing components to be replaced or improved without affecting the rest of the system.

The benchmark construction process can therefore be viewed as a sequence of transformations rather than a monolithic application.

---

# High-Level Architecture

```
                          NCERT Textbooks (PDF)
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │     Extraction Layer      │
                    └──────────────────────────┘
                                   │
                                   ▼
                    Standardized Page Representation
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │      Parsing Layer        │
                    └──────────────────────────┘
                                   │
                                   ▼
                   Hierarchical Textbook Structure
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │    Resource Builder       │
                    └──────────────────────────┘
                                   │
                                   ▼
                      Structured Resource Corpus
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
        Concept Annotation                  Resource Metadata
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │    Query Generation       │
                    └──────────────────────────┘
                                   │
                                   ▼
                       Learner-Oriented Queries
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Candidate Generation      │
                    └──────────────────────────┘
                                   │
                                   ▼
                         Candidate Resource Pool
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Gold Set Generation       │
                    └──────────────────────────┘
                                   │
                                   ▼
                      Proposed Explanation Sets
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Validation Methodology    │
                    └──────────────────────────┘
                                   │
                                   ▼
                          Final Benchmark Dataset
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Baseline Evaluation       │
                    └──────────────────────────┘
                                   │
                                   ▼
                           Retrieval Metrics
```

---

# Stage 1 — Extraction

### Objective

Convert textbook PDFs into a clean page-wise textual representation.

### Input

Raw NCERT textbook PDF

### Output

Page-wise extracted text with optional metadata such as

- equations
- tables
- markdown
- language

The extraction layer does **not** understand textbook structure.

Its sole responsibility is accurate content extraction.

---

# Stage 2 — Parsing

### Objective

Recover the logical organization of the textbook.

The parser transforms a sequence of extracted pages into a structured hierarchy consisting of

- chapters
- sections
- subsections
- paragraph groups

This stage preserves the educational organization of the original textbook.

---

# Stage 3 — Resource Construction

### Objective

Transform textbook content into fine-grained retrieval units.

Rather than indexing entire chapters, EduExplain indexes educational resources.

Examples include

- explanatory passages
- definitions
- worked examples
- activities
- diagrams
- tables
- exercises

Each resource becomes an independently retrievable unit.

---

# Stage 4 — Resource Enrichment

Each resource is further enriched with additional metadata.

These include

- summary
- detailed description
- concept annotations
- textbook metadata
- page information

This stage converts raw educational material into a structured corpus suitable for retrieval.

---

# Stage 5 — Query Generation

Learner-oriented queries are generated from educational concepts.

Rather than generating keyword queries, EduExplain generates realistic learner information needs covering multiple educational intents.

Each query exists in multiple natural-language variants.

---

# Stage 6 — Candidate Generation

For every learner query, a candidate pool of potentially useful educational resources is constructed.

The current implementation combines

- concept-based retrieval
- dense semantic retrieval

The resulting candidate pool defines the search space for explanation-set generation.

---

# Stage 7 — Gold Set Generation

Given

- a learner query
- its candidate resource pool

a Large Language Model proposes one or more explanation sets.

Each explanation set consists of

- selected resources
- an educational explanation
- a rationale describing resource selection

These are considered annotation proposals rather than final benchmark labels.

---

# Stage 8 — Validation

Generated explanation sets undergo validation before becoming part of the benchmark.

Validation verifies

- resource relevance
- explanation completeness
- redundancy
- educational appropriateness
- overall explanation quality

Only validated explanation sets are retained.

---

# Stage 9 — Baseline Evaluation

The benchmark is evaluated using a modular retrieval framework.

Every baseline consists of two independent components.

### Candidate Retrieval

Examples include

- BM25
- Dense Retrieval
- Multimodal Retrieval

### Explanation Set Construction

Examples include

- Top-N Selection
- Maximum Marginal Relevance
- Modality-Capped Selection
- Modality-Aware MMR

This separation enables improvements in retrieval and explanation-set construction to be evaluated independently.

---

# Design Principles

## Modularity

Every stage communicates through standardized intermediate representations.

Replacing one stage should not require modifications to downstream components.

---

## Reproducibility

Every intermediate output is serialized.

Examples include

- extracted pages
- parsed textbook structure
- resource corpus
- generated queries
- candidate pools
- gold explanation sets

This enables every stage to be reproduced independently.

---

## Extensibility

The architecture is designed so that future contributors can

- add new PDF extraction backends
- introduce new retrieval models
- experiment with alternative explanation-set generation strategies
- incorporate additional evaluation metrics

without changing the overall pipeline.

---

# Data Flow Summary

| Stage | Input | Output |
|--------|-------|--------|
| Extraction | PDF | Page-wise text |
| Parsing | Pages | Hierarchical textbook |
| Resource Builder | Hierarchy | Resource corpus |
| Resource Enrichment | Resources | Annotated resources |
| Query Generation | Concepts | Learner queries |
| Candidate Generation | Query | Candidate pool |
| Gold Generation | Query + Candidate Pool | Explanation sets |
| Validation | Explanation sets | Final benchmark |
| Baseline Evaluation | Benchmark | Retrieval metrics |

---

# Guiding Philosophy

EduExplain is designed around a simple principle:

> **The benchmark should evaluate retrieval of educational explanation sets rather than retrieval of individual documents or downstream answer generation.**

Every component of the pipeline exists to support this objective while remaining modular, reproducible, and extensible.
