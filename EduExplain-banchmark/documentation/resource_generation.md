# 03 - Resource Builder

## Purpose

The Resource Builder converts the parsed textbook structure into a collection of fine-grained educational resources that collectively form the retrieval corpus.

Rather than indexing entire chapters or pages, EduExplain indexes semantically meaningful educational units. Each unit is treated as an independent resource and serves as the fundamental retrieval object throughout the benchmark.

Every subsequent stage of the pipeline—including query generation, candidate retrieval, gold explanation generation, and baseline evaluation—operates exclusively on these resources.

---

# Position in the Pipeline

```
Structured Textbook
        │
        ▼
 Resource Builder
        │
        ▼
  Resource Corpus
        │
        ▼
Candidate Retrieval
```

---

# Objectives

The Resource Builder is responsible for

- Converting parsed textbook content into retrieval units.
- Assigning unique resource identifiers.
- Recording textbook metadata.
- Generating concise summaries.
- Generating detailed descriptions.
- Associating educational concepts.
- Producing a standardized resource schema.

The Resource Builder is **not** responsible for

- Query generation.
- Candidate retrieval.
- Gold explanation generation.
- Evaluation.

---

# Why Resources Instead of Chunks?

Traditional retrieval systems often index arbitrary text chunks.

However, educational content contains multiple modalities and pedagogical structures such as

- explanations,
- examples,
- definitions,
- activities,
- figures,
- tables,
- exercises.

Treating every fixed-size chunk equally often separates logically related content or merges unrelated ideas.

EduExplain therefore indexes **educational resources** rather than arbitrary chunks.

Each resource represents one coherent educational unit.

---

# Resource Construction Workflow

```
Parsed Textbook

↓

Identify Educational Unit

↓

Assign Resource Type

↓

Extract Metadata

↓

Generate Summary

↓

Generate Description

↓

Annotate Concepts

↓

Assign Resource ID

↓

Store Resource
```

---

# Resource Types

The corpus currently supports multiple educational resource types.

Typical examples include

- Explanation
- Definition
- Example
- Activity
- Exercise
- Figure
- Table

Additional resource types may be added in the future without modifying downstream components.

---

# Resource Schema

Every resource follows the same schema.

```json
{
    "document_id": "...",
    "resource_id": "...",
    "resource_type": "...",

    "class_no": 12,
    "subject": "Biology",

    "chapter_no": 3,
    "chapter_title": "...",

    "section": "...",
    "subsection": "...",

    "page_start": 25,
    "page_end": 27,

    "text": "...",

    "image_path": null,
    "image_caption": null,

    "summary": "...",
    "description": "...",

    "concepts": [
        ...
    ],

    "concepts_raw": [
        ...
    ],

    "has_equation": false
}
```

---

# Schema Description

## document_id

Unique identifier for the textbook document.

Used to distinguish resources originating from different textbooks.

---

## resource_id

Globally unique identifier for every resource.

This identifier remains stable throughout the pipeline and is used by

- candidate generation,
- gold explanation generation,
- baseline evaluation.

---

## resource_type

Describes the educational role of the resource.

Examples include

- explanation
- example
- figure
- table
- activity

---

## class_no

NCERT class level.

Example

```
11
12
```

---

## subject

Academic subject.

Examples

- Biology
- Chemistry
- Physics

---

## chapter_no

Chapter number within the textbook.

---

## chapter_title

Human-readable chapter title.

---

## section

Section heading containing the resource.

May be null if unavailable.

---

## subsection

Subsection heading containing the resource.

May be null if unavailable.

---

## page_start

First page containing the resource.

---

## page_end

Last page containing the resource.

---

## text

Complete textual content associated with the resource.

This serves as the primary retrieval content.

---

## image_path

Relative path to the associated image.

Null if the resource is purely textual.

---

## image_caption

Caption associated with the image.

Null when unavailable.

---

## summary

Short description of the resource.

Typically one or two sentences.

The summary provides a concise overview and is frequently used during manual inspection and gold-set generation.

---

## description

Detailed description of the educational content.

The description contains richer semantic information than the summary and is intended to improve retrieval quality and provide additional context during benchmark construction.

---

## concepts

Canonical concept annotations associated with the resource.

These concepts form the basis of concept-based retrieval.

---

## concepts_raw

Original concept names before normalization.

These are retained primarily for traceability and debugging.

---

## has_equation

Boolean flag indicating whether the resource contains mathematical equations.

This enables future retrieval methods to incorporate equation-aware indexing if required.

---

# Design Principles

## Atomicity

Each resource should represent one coherent educational idea.

Resources should neither combine unrelated concepts nor unnecessarily fragment a single explanation.

---

## Self-Containment

A resource should contain sufficient information to be understood independently whenever possible.

---

## Stable Identifiers

Resource identifiers must remain unchanged once generated.

Every downstream component relies on these identifiers.

---

## Rich Metadata

Resources store considerably more information than raw text.

This enables

- semantic retrieval,
- concept retrieval,
- multimodal retrieval,
- educational explanation generation.

---

# Output

The Resource Builder produces the complete educational corpus.

```
Book

↓

Resources

↓

all_resources.json
```

This corpus serves as the single source of truth for all downstream benchmark construction stages.

---

# Downstream Usage

The generated resource corpus is subsequently used by

- Concept Annotation
- Query Generation
- Candidate Pool Generation
- Gold Explanation Generation
- Baseline Retrieval
- Benchmark Evaluation

No downstream stage accesses the original textbook PDFs directly.

---

# Key Takeaway

The Resource Builder transforms the parsed textbook hierarchy into a structured collection of educational resources.

These resources represent the fundamental retrieval units of EduExplain and provide a rich, standardized representation of textbook content through unique identifiers, metadata, summaries, descriptions, and concept annotations.
