# 01 - Extraction Layer

## Purpose

The extraction layer is responsible for converting raw NCERT textbook PDFs into a standardized page-wise representation that can be consumed by the parser.

This layer intentionally **does not perform any structural understanding of the textbook**. It is solely responsible for accurately extracting page content while remaining independent of the underlying PDF extraction library.

A key design objective of the extraction layer is **backend independence**. Downstream components should never depend on a specific extraction library such as `pdfplumber` or `Marker`. Instead, all extractors produce the same intermediate representation.

---

# Position in the Pipeline

```
                NCERT Textbook PDF
                        │
                        ▼
            ┌────────────────────────┐
            │   Extraction Backend    │
            └────────────────────────┘
                        │
                        ▼
             Standardized ExtractionResult
                        │
                        ▼
                  Parsing Layer
```

---

# Objectives

The extraction layer is responsible for

- Reading textbook PDFs.
- Preserving page boundaries.
- Preserving reading order.
- Producing clean textual representations.
- Optionally extracting equations.
- Optionally detecting tables.
- Recording extraction metadata.

The extraction layer is **not** responsible for

- Chapter detection.
- Section identification.
- Resource construction.
- Concept annotation.
- Query generation.

These tasks belong to downstream stages.

---

# Design Principles

The extraction layer was designed around four principles.

## 1. Backend Independence

Every extraction backend exposes the same interface.

Whether extraction is performed using

- pdfplumber
- Marker
- Docling
- PyMuPDF
- OCR

the parser receives exactly the same output representation.

This allows extraction libraries to be replaced without modifying downstream code.

---

## 2. Page-Level Abstraction

Extraction produces **pages**, not textbook structure.

A page is the smallest unit exchanged between extraction and parsing.

Any interpretation of headings, chapters, or sections is deferred to the parser.

---

## 3. Reproducibility

Extraction can be performed once and serialized.

Subsequent pipeline stages can operate directly on the serialized output without repeatedly processing PDFs.

This is particularly useful when extraction is performed on HPC clusters or Google Colab.

---

## 4. Extensibility

Adding a new extraction backend should require

- implementing a new extractor class,
- registering it with the factory,

and **nothing else**.

The parser and downstream pipeline should remain unchanged.

---

# Extraction Interface

Every backend returns a common data structure.

```
ExtractionResult

│

├── extraction_backend

├── language

├── has_equations

├── has_tables

└── pages

        ├── page_no

        ├── text

        ├── markdown (optional)

        └── equations (optional)
```

This representation forms the contract between extraction and parsing.

No backend-specific objects should cross this interface.

---

# Supported Extraction Backends

## 1. pdfplumber

Primary purpose:

Lightweight text extraction.

Characteristics

- Fast
- Minimal dependencies
- Suitable for clean digital PDFs

For lower-grade textbooks, extraction proceeds using a standard single-column reading order.

For Class XI and XII textbooks, pages are split into left and right columns before concatenation in reading order.

Basic normalization removes

- repeated characters,
- excessive whitespace,
- printing artefacts,
- page numbering noise.

---

## 2. Marker

Primary purpose:

High-quality structured extraction.

Marker provides significantly richer document structure than plain text extraction.

Additional capabilities include

- equation extraction,
- table detection,
- structured reading order,
- markdown generation.

These richer outputs are converted into the same ExtractionResult representation expected by downstream components.

---

## 3. Precomputed Extraction

Large-scale extraction is computationally expensive.

To avoid repeated execution, extraction outputs may be serialized as JSON.

The Precomputed backend simply reloads these serialized outputs, allowing the remainder of the pipeline to run without any PDF processing.

Typical workflow

```
PDF

↓

Marker

↓

ExtractionResult.json

↓

Precomputed Extractor

↓

Parser
```

---

# Typical Workflow

The extraction phase follows the sequence below.

```
Input PDF

↓

Select Extraction Backend

↓

Extract Individual Pages

↓

Normalize Page Content

↓

Construct ExtractionResult

↓

Serialize (optional)

↓

Pass to Parser
```

---

# Error Handling

Extraction is designed to be tolerant of partial failures.

Whenever possible

- successfully extracted pages are preserved,
- optional metadata is left empty if unavailable,
- downstream stages continue operating on available information.

Extraction failures should not terminate the entire benchmark construction process.

---

# Serialization

Extraction results may be stored as JSON.

Benefits include

- reproducibility,
- faster experimentation,
- separation of extraction and parsing,
- easier debugging,
- portability across execution environments.

This feature is heavily used when extraction is performed remotely.

---

# Responsibilities Summary

| Responsibility | Extraction Layer |
|---------------|------------------|
| Read PDF | ✓ |
| Preserve page order | ✓ |
| Extract page text | ✓ |
| Detect equations | Optional |
| Detect tables | Optional |
| Produce standardized output | ✓ |
| Detect chapters | ✗ |
| Detect sections | ✗ |
| Create resources | ✗ |
| Generate summaries | ✗ |
| Annotate concepts | ✗ |

---

# Input

```
NCERT Textbook PDF
```

---

# Output

```
ExtractionResult

↓

PageData[]

↓

Parser
```

---

# Future Extensions

The current architecture allows additional extraction technologies to be integrated without modifying downstream components.

Examples include

- Docling
- PyMuPDF4LLM
- OCR-based extractors
- Multimodal PDF parsers

Any future backend must satisfy the ExtractionResult interface before entering the remainder of the pipeline.

---

# Key Takeaway

The extraction layer exists solely to transform raw textbook PDFs into a standardized, backend-independent page representation.

It deliberately avoids any semantic interpretation of textbook content, ensuring a clean separation between **document extraction** and **educational structure parsing**.
