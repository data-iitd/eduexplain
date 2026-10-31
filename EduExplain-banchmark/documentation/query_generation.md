# 04 - Query Generation

## Purpose

The Query Generation stage is responsible for constructing realistic learner queries that will later be used to evaluate explanation set retrieval systems.

Unlike conventional information retrieval benchmarks that rely on manually collected search logs, EduExplain generates learner-centric educational queries directly from textbook concepts.

The objective is to create queries that realistically represent the information needs of students while ensuring comprehensive concept coverage across the benchmark.

---

# Position in the Pipeline

```
Resource Corpus
        │
        ▼
Concept Bundles
        │
        ▼
 Query Generation
        │
        ▼
Hierarchical Queries
        │
        ▼
Main Benchmark Queries
        │
        ▼
Robustness Queries
```

---

# Objectives

The Query Generation stage is responsible for

- Constructing learner-oriented educational queries.
- Ensuring balanced concept coverage.
- Supporting multiple educational intents.
- Producing linguistically diverse query formulations.
- Generating both benchmark and robustness query sets.

---

# Input

The query generator receives

- canonical concept annotations,
- concept bundles,
- curriculum metadata.

Concept bundles are automatically constructed from the resource corpus and form the basis for query generation.

---

# Concept Bundles

A concept bundle represents one or more educational concepts that should be addressed by a learner query.

The benchmark currently supports

- Single-concept bundles
- Two-concept bundles
- Three-concept bundles

Each generated query is associated with exactly one concept bundle.

---

# Query Types

To simulate different learner information needs, EduExplain generates multiple query types.

Current query categories include

- Explanation
- Why / How
- Example
- Revision
- Practice

These categories encourage retrieval systems to retrieve explanation sets suitable for different educational contexts rather than optimizing only for factual questions.

---

# Query Variants

Each generated query is expressed using multiple natural-language formulations.

Current variants include

- Search
- Chat
- Learning Context

For example, the same educational intent may appear as

```
Search

"What is meiosis?"
```

```
Chat

"Can you explain what meiosis is?"
```

```
Learning Context

"I was reading about cell division but I'm confused about meiosis. Could you explain it?"
```

These variants improve linguistic diversity while preserving the underlying information need.

---

# Hierarchical Query Generation

The primary output of this stage is a hierarchical query file.

Conceptually

```
Concept Bundle

↓

Query Type

↓

Query Variants
```

Each bundle therefore contains

- multiple educational intents,
- multiple linguistic formulations.

This intermediate representation is retained for reproducibility.

---

# Benchmark Construction

The hierarchical queries are subsequently divided into two benchmark subsets.

## Main Benchmark

One variant is randomly selected from each generated query.

These form the primary benchmark used during evaluation.

---

## Robustness Benchmark

The remaining variants are retained separately.

These queries enable evaluation of retrieval robustness under different natural-language formulations without changing the underlying educational intent.

---

# Balancing Strategy

The benchmark is designed to provide broad coverage across

- concept combinations,
- educational intents,
- linguistic styles.

The final benchmark contains a balanced mixture of

- single-concept queries,
- two-concept queries,
- three-concept queries,

distributed across all supported query types.

---

# Output

The query generation stage produces three primary artifacts.

```
Hierarchical Queries

↓

Main Queries

↓

Robustness Queries
```

These files are subsequently consumed by the candidate generation stage.

---

# Design Principles

## Educational Realism

Queries should resemble genuine learner questions rather than keyword searches.

---

## Diversity

Multiple formulations of the same information need are retained to improve robustness evaluation.

---

## Balanced Coverage

No individual concept or query type should dominate the benchmark.

---

## Reproducibility

The hierarchical query representation is preserved, allowing the benchmark to be regenerated or extended without repeating the generation process.

---

# Downstream Usage

The generated benchmark queries are used exclusively by the Candidate Generation stage.

Each query serves as the starting point for constructing a candidate pool of educational resources.

---

# Key Takeaway

The Query Generation stage transforms concept bundles into realistic learner queries while ensuring balanced concept coverage, multiple educational intents, and linguistic diversity.

The resulting benchmark consists of

- a primary evaluation set, and
- a robustness evaluation set,

both derived from a common hierarchical query representation.
