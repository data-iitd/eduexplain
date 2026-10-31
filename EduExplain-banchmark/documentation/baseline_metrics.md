# 06 - Baseline Evaluation Pipeline

## Purpose

The objective of the baseline evaluation pipeline is to establish reproducible reference results for the EduExplain benchmark.

Unlike traditional document retrieval benchmarks, EduExplain evaluates **Explanation Set Retrieval (ESR)**, where the goal is to retrieve a compact and complementary set of educational resources that collectively answer a learner query.

The baseline pipeline therefore consists of two independent components:

1. **Candidate Retrieval**
2. **Explanation Set Construction**

This modular design allows retrieval models and explanation-set construction algorithms to be evaluated independently.

---

# Position in the Pipeline

```
Benchmark Queries
        │
        ▼
 Candidate Retrieval
        │
        ▼
 Ranked Candidate Resources
        │
        ▼
Explanation Set Construction
        │
        ▼
 Predicted Explanation Set
        │
        ▼
Evaluation Metrics
```

---

# Evaluation Pipeline

```
Learner Query

↓

Retriever

↓

Ranked Resource List

↓

Explanation Set Selector

↓

Predicted Explanation Set

↓

Compare with Gold Explanation Sets

↓

Compute Evaluation Metrics
```

---

# Stage 1 — Candidate Retrieval

## Objective

Retrieve a ranked list of educational resources that are potentially useful for answering the learner query.

The retriever operates over the complete resource corpus generated during benchmark construction.

---

## Supported Retrieval Models

The baseline framework is designed to support multiple retrieval paradigms.

### BM25

Lexical retrieval based on exact term matching.

Characteristics

- Sparse retrieval
- Fast indexing
- Strong keyword matching
- Limited semantic understanding

---

### Dense Retrieval

Semantic retrieval using dense embeddings.

Examples

- BGE-M3
- DPR
- E5

Characteristics

- Embedding-based retrieval
- Captures semantic similarity
- Better handling of paraphrases
- Higher computational cost

---

### Multimodal Retrieval

Retrieval using both textual and visual representations.

Examples

- OpenCLIP
- SigLIP

Characteristics

- Joint text-image embedding space
- Supports figures, diagrams, and images
- Useful for multimodal educational resources

---

# Retrieval Output

Every retrieval model produces

```
Query

↓

Ranked Resource List

↓

Top-K Resources
```

The ranked list is then passed to the explanation-set construction stage.

---

# Stage 2 — Explanation Set Construction

## Objective

Transform a ranked resource list into a compact explanation set.

Unlike conventional retrieval systems that simply return the top-ranked documents, EduExplain evaluates whether the retrieved resources collectively provide a complete educational explanation.

---

# Supported Selection Strategies

## Top-N

The simplest baseline.

```
Ranked Resources

↓

Top N Resources

↓

Explanation Set
```

No diversity constraints are applied.

---

## Maximum Marginal Relevance (MMR)

MMR balances

- relevance
- diversity

Resources are selected iteratively to maximize relevance while minimizing redundancy.

This encourages explanation sets containing complementary educational information.

---

## Modality-Capped Selection

Ensures that no single resource modality dominates the explanation set.

For example

```
Maximum

2 Explanations

1 Figure

1 Table
```

This encourages multimodal explanations.

---

## Modality-Aware MMR

Extension of standard MMR.

Balances

- relevance
- redundancy
- modality diversity

This strategy encourages explanation sets containing complementary information from multiple educational modalities.

---

# Predicted Explanation Set

The explanation-set selector outputs

```
Resource 1

Resource 2

Resource 3

...
```

This predicted explanation set is compared directly against the benchmark gold sets.

---

# Evaluation Metrics

The benchmark supports several complementary evaluation metrics.

---

## Recall@K

Measures whether the retriever successfully retrieves relevant resources within the top K ranked results.

Higher Recall@K indicates stronger retrieval performance.

---

## Gold Set Recall

Measures the proportion of gold explanation resources successfully retrieved.

This metric evaluates how completely a predicted explanation set covers the benchmark gold set.

---

## Gold Set Precision

Measures the proportion of retrieved resources that belong to the gold explanation set.

Higher precision indicates lower redundancy.

---

## Gold Set F1

Harmonic mean of

- Gold Set Recall
- Gold Set Precision

Provides a balanced measure of explanation-set retrieval quality.

---

## Concept Coverage

Measures how well the retrieved explanation set covers the educational concepts represented in the gold explanation sets.

This metric rewards retrieval systems that retrieve conceptually complete explanations rather than isolated resources.

---

# Benchmark Protocol

For every learner query

```
Query

↓

Retriever

↓

Ranked Resources

↓

Explanation Set Construction

↓

Predicted Explanation Set

↓

Compare Against Gold Sets

↓

Compute Metrics
```

Final benchmark scores are obtained by averaging performance across all benchmark queries.

---

# Extending the Baseline

The modular architecture allows future contributors to evaluate additional retrieval models without modifying the benchmark itself.

Examples include

- ColBERT
- SPLADE
- Contriever
- ModernBERT
- Hybrid Retrieval

Similarly, new explanation-set construction algorithms may be incorporated without changing the evaluation pipeline.

---

# Responsibilities Summary

| Stage | Responsibility |
|---------|----------------|
| Candidate Retrieval | Retrieve relevant educational resources |
| Ranking | Produce ranked resource list |
| Explanation Set Construction | Select complementary resources |
| Evaluation | Compare predicted and gold explanation sets |
| Metrics | Compute benchmark performance |

---

# Key Takeaway

The EduExplain baseline framework separates **retrieval** from **explanation-set construction**, allowing each component to be studied independently.

Retrievers focus on identifying relevant educational resources, while explanation-set construction algorithms determine which combination of retrieved resources best forms a coherent educational explanation.

This modular evaluation protocol enables fair comparison of retrieval models, explanation-set selection strategies, and future multimodal retrieval approaches while maintaining a common benchmark and evaluation methodology.
