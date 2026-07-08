# Medical Deep Agent

An **artifact-driven AI research agent** for **Medical AI literature review**, **research gap discovery**, and **evidence-grounded research idea generation**.

Instead of directly summarizing retrieved papers, Medical Deep Agent transforms academic evidence into structured research artifacts before performing multi-stage reasoning to identify research gaps and generate research directions.

---

# Motivation

Medical AI researchers often spend significant time searching papers, comparing methods, tracking citations, and identifying research gaps manually.

Traditional LLM-based research assistants mainly perform search and summarization, making it difficult to:

* Compare evidence systematically
* Discover field-level research gaps
* Analyze tradeoffs across methods
* Generate evidence-grounded research ideas

Medical Deep Agent addresses these challenges through an **artifact-driven reasoning workflow** that separates retrieval, evidence structuring, reasoning, and report generation.

---

# Key Features

* 🔍 Adaptive query planning
* 📚 PubMed + arXiv academic retrieval
* 📑 Automatic Evidence Matrix generation
* 📝 Structured Source Notes
* 📖 Citation Registry
* 🧠 Two-stage gap reasoning
* ⚖️ Cross-paper tradeoff analysis
* 💡 Evidence-grounded research idea generation
* ✅ Citation-grounded research reports

---

# Workflow

```text
User Query
      │
      ▼
Adaptive Query Planning
      │
      ▼
Academic Retrieval
(PubMed / arXiv / Web)
      │
      ▼
raw_evidence.json
      │
      ▼
Artifact Builder
      │
      ├── evidence_matrix.md
      ├── citation_registry.md
      └── research-notes/source_notes.md
      │
      ▼
Gap Reasoning Engine
      │
      ├── Semantic Clustering
      ├── LLM Second-pass Conceptual Merge
      ├── Tradeoff Analysis
      └── Validated Research Gaps
      │
      ▼
gap_analysis.md
idea_bank.md
gap_clusters.json
      │
      ▼
Final Research Report
```

---

# Core Components

## Adaptive Query Planning

Automatically classifies user intent and selects the appropriate research workflow.

Supported modes:

* Direct Question
* Literature Review
* Research Advisor
* Report Refinement (optional)

---

## Artifact Builder

Transforms raw academic search results into reusable research artifacts.

Generated artifacts:

* Evidence Matrix
* Citation Registry
* Source Notes
* Raw Evidence

These structured artifacts become the input for downstream reasoning instead of allowing the LLM to reason directly over raw search results.

---

## Gap Reasoning Engine

A custom reasoning engine designed for research gap discovery.

### Stage 1 — Semantic Clustering

Extracts:

* author-stated limitations
* research motivations
* claimed gaps

and groups similar evidence using embeddings.

### Stage 2 — Conceptual Reasoning

An LLM performs a second reasoning pass to:

* merge conceptually related clusters
* identify hidden tradeoffs
* infer validated research gaps
* generate evidence-grounded research ideas

Unlike embedding clustering, this stage reasons at the **concept level** rather than only semantic similarity.

---

## Tradeoff Analysis

Rather than listing limitations independently, the system identifies tradeoffs across different research directions.

Examples include:

* Performance ↔ Efficiency
* Privacy ↔ Utility
* Generalization ↔ Specialization
* Annotation Cost ↔ Supervision Quality

These tradeoffs are used to infer higher-value research gaps.

---

# Generated Artifacts

| Artifact                         | Purpose                                  |
| -------------------------------- | ---------------------------------------- |
| `raw_evidence.json`              | Raw retrieved academic evidence          |
| `evidence_matrix.md`             | Structured comparison across papers      |
| `citation_registry.md`           | Source tracking and citation mapping     |
| `research-notes/source_notes.md` | Detailed notes extracted from each paper |
| `gap_clusters.json`              | Clustered limitations and tradeoff graph |
| `gap_analysis.md`                | Validated research gaps                  |
| `idea_bank.md`                   | Evidence-grounded research ideas         |
| `final_report.md`                | Final synthesized report                 |

---

# Tech Stack

* LangGraph Deep Agent
* LangSmith Studio
* Python
* OpenAI GPT Models
* PubMed Retrieval
* arXiv Retrieval
* Tavily Search
* Markdown Artifacts
* JSON Artifacts

---

# Repository Structure

```text
medical-deep-agent/

├── main.py
├── artifact_builder_tool.py
├── gap_reasoning_tool.py
├── final_report_tool.md
├── pyproject.toml
├── langgraph.json
├── README.md
├── .env
├── .gitignore
│
├── skills/
│   └── deep-research/
│       └── SKILL.md
│
├── research-notes/
│   └── source_notes.md
│
├── outputs/
│
├── raw_evidence.json
├── evidence_matrix.md
├── citation_registry.md
├── gap_clusters.json
├── gap_analysis.md
├── idea_bank.md
├── final_report.md
└── langgraph_report.md
```

---

# Example Tasks

* Literature review for Medical AI
* Research gap discovery
* Paper comparison
* Method comparison
* Dataset comparison
* Research idea generation
* Academic report generation

---

# Current Limitations

* Gap reasoning quality depends on evidence extraction quality.
* Clustering granularity can still be improved for deeper conceptual reasoning.
* Research gap ranking is currently heuristic-based.
* Human verification is recommended for high-stakes academic conclusions.

---

# Future Work

* Improve prompt efficiency
* Enhance conceptual clustering algorithms
* Introduce automated evaluation benchmarks
* Human-in-the-loop validation
* Support multi-agent collaborative research

---

# Author

**Ai Phuong Nguyen**

Medical AI • Deep Learning • LLM Agents • Research Automation

