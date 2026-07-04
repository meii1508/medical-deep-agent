import os
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

# PROJECT_ROOT = Path("/home/aiphuongnguyen/medical-deep-agent")

# OPENAI_MODEL = os.getenv("GAP_REASONING_MODEL", "gpt-4.1-mini")
# EMBEDDING_MODEL = os.getenv("GAP_EMBEDDING_MODEL", "text-embedding-3-small")

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# def _read_file(path: str) -> str:
#     file_path = PROJECT_ROOT / path
#     if not file_path.exists():
#         return ""
#     return file_path.read_text(encoding="utf-8")


# def _write_file(path: str, content: str) -> None:
#     file_path = PROJECT_ROOT / path
#     file_path.parent.mkdir(parents=True, exist_ok=True)
#     file_path.write_text(content, encoding="utf-8")


# def _safe_json_loads(text: str) -> Any:
#     """
#     Try to parse JSON even if the model wraps it in markdown.
#     """
#     text = text.strip()

#     if text.startswith("```"):
#         text = re.sub(r"^```(?:json)?", "", text)
#         text = re.sub(r"```$", "", text).strip()

#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
#         if match:
#             return json.loads(match.group(1))
#         raise


# def _llm_json(system_prompt: str, user_prompt: str) -> Any:
#     response = client.chat.completions.create(
#         model=OPENAI_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         temperature=0.1,
#     )
#     return _safe_json_loads(response.choices[0].message.content)


# def _embed_texts(texts: list[str]) -> np.ndarray:
#     response = client.embeddings.create(
#         model=EMBEDDING_MODEL,
#         input=texts,
#     )
#     embeddings = [item.embedding for item in response.data]
#     return np.array(embeddings, dtype=np.float32)


# def extract_author_stated_limitations(
#     evidence_matrix: str,
#     source_notes: str,
# ) -> list[dict]:
#     """
#     Extract atomic author-stated limitations from existing artifacts.
#     This stage tries to prevent generic hallucinated limitations.
#     """
#     system_prompt = """
# You extract author-stated limitations from academic evidence.

# Rules:
# - Extract only limitations supported by the provided evidence.
# - Do not invent generic medical AI limitations.
# - Do not write "limited external validation", "limited domain shift", or "small dataset" unless explicitly supported.
# - Break compound limitations into atomic limitation statements.
# - Keep evidence quotes short.
# - Return valid JSON only.
# """

#     user_prompt = f"""
# Extract atomic author-stated limitations from the evidence below.

# Return JSON array with this schema:
# [
#   {{
#     "paper_id": "S1 or paper title if no ID",
#     "paper_title": "...",
#     "limitation_text": "...",
#     "evidence_quote": "...",
#     "limitation_type_hint": "...",
#     "is_author_stated": true,
#     "confidence": "high|medium|low"
#   }}
# ]

# Evidence Matrix:
# ================
# {evidence_matrix}

# Source Notes:
# =============
# {source_notes}
# """

#     result = _llm_json(system_prompt, user_prompt)

#     if isinstance(result, dict) and "limitations" in result:
#         result = result["limitations"]

#     if not isinstance(result, list):
#         return []

#     cleaned = []
#     for item in result:
#         if not isinstance(item, dict):
#             continue
#         if not item.get("limitation_text"):
#             continue
#         cleaned.append(item)

#     return cleaned


# def cluster_limitations(
#     limitations: list[dict],
#     distance_threshold: float = 0.28,
# ) -> list[dict]:
#     """
#     Cluster limitations in embedding space.

#     Lower distance_threshold = stricter clusters.
#     Higher distance_threshold = broader clusters.

#     Since sklearn uses cosine distance = 1 - cosine similarity:
#     distance_threshold 0.28 roughly means cosine similarity around 0.72.
#     """
#     if not limitations:
#         return []

#     if len(limitations) == 1:
#         return [{
#             "cluster_id": "C1",
#             "items": limitations,
#         }]

#     texts = [
#         f"{x.get('limitation_text', '')} {x.get('limitation_type_hint', '')}"
#         for x in limitations
#     ]

#     embeddings = _embed_texts(texts)

#     clustering = AgglomerativeClustering(
#         n_clusters=None,
#         metric="cosine",
#         linkage="average",
#         distance_threshold=distance_threshold,
#     )

#     labels = clustering.fit_predict(embeddings)

#     clusters = {}
#     for label, item in zip(labels, limitations):
#         clusters.setdefault(int(label), []).append(item)

#     output = []
#     for idx, (_, items) in enumerate(clusters.items(), start=1):
#         output.append({
#             "cluster_id": f"C{idx}",
#             "items": items,
#         })

#     return output


# def label_clusters_and_infer_gaps(clusters: list[dict]) -> dict:
#     """
#     LLM labels each limitation cluster and infers validated research gaps.
#     """
#     system_prompt = """
# You are a research-gap reasoning module for medical AI literature review.

# Your job:
# 1. Label clusters of author-stated limitations.
# 2. Infer validated research gaps.
# 3. Avoid unsupported generic gaps.

# Rules:
# - A research gap must be supported by cluster evidence.
# - Prefer gaps supported by multiple papers.
# - A single-paper gap is allowed only if the evidence explicitly says it is underexplored, missing, or unresolved.
# - Preserve the authors' logic.
# - Separate paper limitations from field-level research gaps.
# - Return valid JSON only.
# """

#     user_prompt = f"""
# Here are clusters of author-stated limitations.

# Infer validated research gaps.

# For each cluster, produce:
# - cluster_label
# - limitation_pattern
# - supporting_papers
# - evidence_summary
# - possible_field_gap
# - confidence

# For each validated research gap, produce:
# - gap
# - why_it_matters
# - supporting_clusters
# - supporting_papers
# - confidence
# - possible_research_idea
# - required_data
# - feasibility
# - risks

# Return JSON with schema:
# {{
#   "clusters": [
#     {{
#       "cluster_id": "C1",
#       "cluster_label": "...",
#       "limitation_pattern": "...",
#       "supporting_papers": ["..."],
#       "evidence_summary": "...",
#       "possible_field_gap": "...",
#       "confidence": "high|medium|low"
#     }}
#   ],
#   "validated_research_gaps": [
#     {{
#       "gap": "...",
#       "why_it_matters": "...",
#       "supporting_clusters": ["C1"],
#       "supporting_papers": ["..."],
#       "confidence": "high|medium|low",
#       "possible_research_idea": "...",
#       "required_data": "...",
#       "feasibility": "high|medium|low",
#       "risks": "..."
#     }}
#   ]
# }}

# Clusters:
# =========
# {json.dumps(clusters, ensure_ascii=False, indent=2)}
# """

#     result = _llm_json(system_prompt, user_prompt)

#     if not isinstance(result, dict):
#         return {
#             "clusters": [],
#             "validated_research_gaps": [],
#         }

#     return {
#         "clusters": result.get("clusters", []),
#         "validated_research_gaps": result.get("validated_research_gaps", []),
#     }


# def _markdown_table(rows: list[dict], columns: list[str]) -> str:
#     if not rows:
#         return "_No rows._"

#     header = "| " + " | ".join(columns) + " |"
#     sep = "| " + " | ".join(["---"] * len(columns)) + " |"
#     lines = [header, sep]

#     for row in rows:
#         values = []
#         for col in columns:
#             value = str(row.get(col, "")).replace("\n", " ").strip()
#             values.append(value)
#         lines.append("| " + " | ".join(values) + " |")

#     return "\n".join(lines)


# def write_gap_outputs(
#     limitations: list[dict],
#     clusters: list[dict],
#     reasoning: dict,
# ) -> None:
#     """
#     Write:
#     - gap_clusters.json
#     - gap_analysis.md
#     - idea_bank.md
#     """

#     _write_file(
#         "gap_clusters.json",
#         json.dumps(
#             {
#                 "limitations": limitations,
#                 "raw_clusters": clusters,
#                 "reasoning": reasoning,
#             },
#             ensure_ascii=False,
#             indent=2,
#         ),
#     )

#     cluster_rows = []
#     for c in reasoning.get("clusters", []):
#         cluster_rows.append({
#             "Cluster ID": c.get("cluster_id", ""),
#             "Cluster Label": c.get("cluster_label", ""),
#             "Limitation Pattern": c.get("limitation_pattern", ""),
#             "Supporting Papers": ", ".join(c.get("supporting_papers", [])),
#             "Possible Field Gap": c.get("possible_field_gap", ""),
#             "Confidence": c.get("confidence", ""),
#         })

#     gap_rows = []
#     for g in reasoning.get("validated_research_gaps", []):
#         gap_rows.append({
#             "Research Gap": g.get("gap", ""),
#             "Why It Matters": g.get("why_it_matters", ""),
#             "Supporting Papers": ", ".join(g.get("supporting_papers", [])),
#             "Confidence": g.get("confidence", ""),
#         })

#     idea_rows = []
#     for g in reasoning.get("validated_research_gaps", []):
#         idea_rows.append({
#             "Gap": g.get("gap", ""),
#             "Proposed Idea": g.get("possible_research_idea", ""),
#             "Required Data": g.get("required_data", ""),
#             "Feasibility": g.get("feasibility", ""),
#             "Risks": g.get("risks", ""),
#             "Evidence": ", ".join(g.get("supporting_papers", [])),
#         })

#     gap_md = f"""# Gap Analysis

# ## Method

# This gap analysis was generated using a hybrid reasoning engine:

# 1. Extract author-stated limitations from existing evidence.
# 2. Embed atomic limitation statements.
# 3. Cluster semantically similar limitations.
# 4. Use an LLM to label clusters and infer validated research gaps.

# No new papers are introduced by this tool. It reasons over existing artifacts.

# ---

# ## Limitation Clusters

# {_markdown_table(cluster_rows, [
#     "Cluster ID",
#     "Cluster Label",
#     "Limitation Pattern",
#     "Supporting Papers",
#     "Possible Field Gap",
#     "Confidence",
# ])}

# ---

# ## Validated Research Gaps

# {_markdown_table(gap_rows, [
#     "Research Gap",
#     "Why It Matters",
#     "Supporting Papers",
#     "Confidence",
# ])}
# """

#     idea_md = f"""# Idea Bank

# Research ideas are generated only from validated research gaps.

# {_markdown_table(idea_rows, [
#     "Gap",
#     "Proposed Idea",
#     "Required Data",
#     "Feasibility",
#     "Risks",
#     "Evidence",
# ])}
# """

#     _write_file("gap_analysis.md", gap_md)
#     _write_file("idea_bank.md", idea_md)

# @tool
# def gap_reasoning_tool(
#     evidence_matrix_path: str = "evidence_matrix.md",
#     source_notes_path: str = "research-notes/source_notes.md",
#     distance_threshold: float = 0.28,
# ) -> dict:
#     """
#     Hybrid gap reasoning tool.

#     Use this after evidence_matrix.md and source notes exist.

#     Pipeline:
#     1. Extract author-stated limitations.
#     2. Cluster limitations using embeddings.
#     3. Label clusters with LLM.
#     4. Infer validated research gaps.
#     5. Write gap_analysis.md and idea_bank.md.

#     This tool should not be used as a search tool.
#     It does not retrieve new papers.
#     It reasons over existing artifacts.
#     """
#     evidence_matrix = _read_file(evidence_matrix_path)
#     source_notes = _read_file(source_notes_path)

#     if not evidence_matrix and not source_notes:
#         return {
#             "ok": False,
#             "error": "No evidence_matrix.md or source_notes.md found. Build evidence artifacts first.",
#         }

#     limitations = extract_author_stated_limitations(
#         evidence_matrix=evidence_matrix,
#         source_notes=source_notes,
#     )

#     if not limitations:
#         return {
#             "ok": False,
#             "error": "No author-stated limitations could be extracted from existing artifacts.",
#         }

#     raw_clusters = cluster_limitations(
#         limitations=limitations,
#         distance_threshold=distance_threshold,
#     )

#     reasoning = label_clusters_and_infer_gaps(raw_clusters)

#     write_gap_outputs(
#         limitations=limitations,
#         clusters=raw_clusters,
#         reasoning=reasoning,
#     )

#     return {
#         "ok": True,
#         "num_limitations": len(limitations),
#         "num_clusters": len(raw_clusters),
#         "num_validated_gaps": len(reasoning.get("validated_research_gaps", [])),
#         "outputs": [
#             "gap_clusters.json",
#             "gap_analysis.md",
#             "idea_bank.md",
#         ],
#         "summary": reasoning,
#     }

PROJECT_ROOT = Path("/home/aiphuongnguyen/medical-deep-agent")

OPENAI_MODEL = os.getenv("GAP_REASONING_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("GAP_EMBEDDING_MODEL", "text-embedding-3-small")


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to .env or export it before running.")
    return OpenAI(api_key=api_key)


def _read_file(path: str) -> str:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def _write_file(path: str, content: str) -> None:
    file_path = PROJECT_ROOT / path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def _safe_json_loads(text: str) -> Any:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def _llm_json(system_prompt: str, user_prompt: str) -> Any:
    client = get_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return _safe_json_loads(response.choices[0].message.content)


def _embed_texts(texts: list[str]) -> np.ndarray:
    client = get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype=np.float32)


def extract_author_stated_limitations(
    evidence_matrix: str,
    source_notes: str,
    raw_evidence: str = "",
) -> list[dict]:
    """
    Extract atomic author-stated limitations from existing artifacts.
    """
    system_prompt = """
You extract author-stated limitations from academic evidence.

Rules:
- Extract only limitations supported by the provided evidence.
- Do not invent generic medical AI limitations.
- Do not write "limited external validation", "limited domain shift", or "small dataset" unless explicitly supported.
- Break compound limitations into atomic limitation statements.
- Include the paper/source ID if available.
- Keep evidence quotes short.
- Return valid JSON only.
"""

    user_prompt = f"""
Extract atomic author-stated limitations from the evidence below.

Return JSON array with this schema:
[
  {{
    "paper_id": "S1 or paper title if no ID",
    "paper_title": "...",
    "limitation_text": "...",
    "evidence_quote": "...",
    "limitation_type_hint": "...",
    "is_author_stated": true,
    "confidence": "high|medium|low"
  }}
]

Evidence Matrix:
================
{evidence_matrix}

Source Notes:
=============
{source_notes}

Raw Evidence, optional:
======================
{raw_evidence}
"""

    result = _llm_json(system_prompt, user_prompt)

    if isinstance(result, dict) and "limitations" in result:
        result = result["limitations"]

    if not isinstance(result, list):
        return []

    cleaned = []
    for item in result:
        if not isinstance(item, dict):
            continue
        if not item.get("limitation_text"):
            continue
        cleaned.append(item)

    return cleaned


def cluster_limitations(
    limitations: list[dict],
    distance_threshold: float = 0.38,
) -> list[dict]:
    """
    First-pass embedding clustering.

    Higher distance_threshold = broader clusters.
    
    """
    if not limitations:
        return []

    if len(limitations) == 1:
        return [{"cluster_id": "C1", "items": limitations}]

    texts = [
        f"{x.get('limitation_text', '')} {x.get('limitation_type_hint', '')}"
        for x in limitations
    ]

    embeddings = _embed_texts(texts)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )

    labels = clustering.fit_predict(embeddings)

    clusters = {}
    for label, item in zip(labels, limitations):
        clusters.setdefault(int(label), []).append(item)

    output = []
    for idx, (_, items) in enumerate(clusters.items(), start=1):
        output.append({
            "cluster_id": f"C{idx}",
            "items": items,
        })

    return output


def merge_clusters_conceptually(raw_clusters: list[dict]) -> dict:
    """
    Second-pass LLM merge.

    Embedding clustering catches surface similarity.
    This step merges conceptually related clusters that form a shared gap,
    tradeoff, or hidden assumption.
    """
    system_prompt = """
You are a conceptual cluster merger for academic research-gap analysis.

You receive first-pass limitation clusters. These clusters may be too fragmented.

Your job:
1. Merge clusters that are conceptually related.
2. Preserve evidence and supporting paper IDs.
3. Explain the merge logic.
4. Do not invent new limitations.
5. Do not merge unrelated clusters just to reduce the number.
6. Return valid JSON only.

Merge clusters if they share:
- the same underlying variable,
- the same unresolved tradeoff,
- the same evaluation weakness,
- the same deployment constraint,
- the same hidden assumption,
- or together form one research gap.

Examples:
- task order + initial task quality + representation quality can be merged if the common issue is task-sequence-sensitive representation learning.
- rehearsal effectiveness + privacy constraint can be linked as a tradeoff, not necessarily merged as the same limitation.
"""

    user_prompt = f"""
First-pass clusters:
{json.dumps(raw_clusters, ensure_ascii=False, indent=2)}

Return JSON with this schema:
{{
  "merged_clusters": [
    {{
      "merged_cluster_id": "MC1",
      "cluster_label": "...",
      "merged_from": ["C1", "C3"],
      "supporting_papers": ["S1", "S3"],
      "limitation_pattern": "...",
      "merge_rationale": "...",
      "items": [
        {{
          "paper_id": "...",
          "paper_title": "...",
          "limitation_text": "...",
          "evidence_quote": "..."
        }}
      ]
    }}
  ],
  "unmerged_clusters": [
    {{
      "merged_cluster_id": "MC2",
      "cluster_label": "...",
      "merged_from": ["C2"],
      "supporting_papers": ["S2"],
      "limitation_pattern": "...",
      "merge_rationale": "Kept separate because ...",
      "items": [...]
    }}
  ]
}}
"""

    result = _llm_json(system_prompt, user_prompt)

    merged = result.get("merged_clusters", []) if isinstance(result, dict) else []
    unmerged = result.get("unmerged_clusters", []) if isinstance(result, dict) else []

    all_clusters = merged + unmerged

    # Normalize IDs if missing
    for idx, c in enumerate(all_clusters, start=1):
        c.setdefault("merged_cluster_id", f"MC{idx}")
        c.setdefault("supporting_papers", [])
        c.setdefault("merged_from", [])
        c.setdefault("items", [])

    return {
        "merged_clusters": all_clusters
    }


def build_tradeoff_graph(merged_clusters: list[dict]) -> dict:
    """
    Build an intermediate contradiction/tradeoff graph before gap inference.
    """
    system_prompt = """
You are a contradiction and tradeoff mining module for academic research-gap discovery.

You receive merged limitation clusters.

Your job:
1. Identify hidden contradictions, tensions, or tradeoffs across clusters.
2. Explain which assumptions conflict.
3. Tie every tradeoff to supporting clusters and papers.
4. Do not invent claims not supported by clusters.
5. Return valid JSON only.

Examples of tradeoff patterns often seen in medical AI literature include:

- performance vs efficiency
- robustness vs adaptability
- generalization vs specialization
- privacy vs utility
- annotation cost vs supervision quality
- interpretability vs model complexity
- benchmark control vs clinical realism

These examples are illustrative only.
Do not try to force every cluster to fit these categories.
If evidence suggests a novel tradeoff, prefer the evidence.
"""

    user_prompt = f"""
Merged clusters:
{json.dumps(merged_clusters, ensure_ascii=False, indent=2)}

Return JSON:
{{
  "tradeoffs": [
    {{
      "tradeoff_id": "T1",
      "tradeoff_label": "...",
      "side_a": "...",
      "side_b": "...",
      "supporting_clusters": ["MC1", "MC2"],
      "supporting_papers": ["S1", "S2"],
      "hidden_assumption": "...",
      "why_it_matters": "...",
      "confidence": "high|medium|low"
    }}
  ]
}}
"""

    result = _llm_json(system_prompt, user_prompt)

    if not isinstance(result, dict):
        return {"tradeoffs": []}

    return {
        "tradeoffs": result.get("tradeoffs", [])
    }


def infer_gaps_from_clusters_and_tradeoffs(
    merged_clusters: list[dict],
    tradeoff_graph: dict,
) -> dict:
    """
    Infer validated research gaps from merged clusters and tradeoff graph.
    """
    system_prompt = """
You are a research-gap reasoning module for medical AI literature review.

Your job:
1. Label merged limitation patterns.
2. Use tradeoffs to infer latent field-level research gaps.
3. Generate feasible research ideas.
4. Avoid unsupported generic gaps.
5. Return valid JSON only.

Rules:
- A research gap must be supported by cluster evidence or tradeoff evidence.
- Prefer gaps supported by multiple papers or multiple clusters.
- A single-paper gap is allowed only if the evidence explicitly states the issue is underexplored, missing, or unresolved.
- Separate paper limitations from field-level research gaps.
- Preserve the authors' logic.
- Research ideas must be specific and testable.
"""

    user_prompt = f"""
Merged clusters:
{json.dumps(merged_clusters, ensure_ascii=False, indent=2)}

Tradeoff graph:
{json.dumps(tradeoff_graph, ensure_ascii=False, indent=2)}

Infer validated research gaps and research ideas.

Return JSON:
{{
  "clusters": [
    {{
      "cluster_id": "MC1",
      "cluster_label": "...",
      "limitation_pattern": "...",
      "supporting_papers": ["..."],
      "evidence_summary": "...",
      "possible_field_gap": "...",
      "confidence": "high|medium|low"
    }}
  ],
  "tradeoffs": [
    {{
      "tradeoff_id": "T1",
      "tradeoff_label": "...",
      "supporting_papers": ["..."],
      "latent_gap": "...",
      "confidence": "high|medium|low"
    }}
  ],
  "validated_research_gaps": [
    {{
      "gap": "...",
      "why_it_matters": "...",
      "supporting_clusters": ["MC1"],
      "supporting_tradeoffs": ["T1"],
      "supporting_papers": ["..."],
      "confidence": "high|medium|low",
      "possible_research_idea": "...",
      "required_data": "...",
      "feasibility": "high|medium|low",
      "risks": "..."
    }}
  ]
}}
"""

    result = _llm_json(system_prompt, user_prompt)

    if not isinstance(result, dict):
        return {
            "clusters": [],
            "tradeoffs": [],
            "validated_research_gaps": [],
        }

    return {
        "clusters": result.get("clusters", []),
        "tradeoffs": result.get("tradeoffs", []),
        "validated_research_gaps": result.get("validated_research_gaps", []),
    }


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]

    for row in rows:
        values = []
        for col in columns:
            value = str(row.get(col, "")).replace("\n", " ").strip()
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_gap_outputs(
    limitations: list[dict],
    raw_clusters: list[dict],
    merged_clusters: list[dict],
    tradeoff_graph: dict,
    reasoning: dict,
) -> None:
    _write_file(
        "gap_clusters.json",
        json.dumps(
            {
                "limitations": limitations,
                "raw_clusters": raw_clusters,
                "merged_clusters": merged_clusters,
                "tradeoff_graph": tradeoff_graph,
                "reasoning": reasoning,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    cluster_rows = []
    for c in reasoning.get("clusters", []):
        cluster_rows.append({
            "Cluster ID": c.get("cluster_id", ""),
            "Cluster Label": c.get("cluster_label", ""),
            "Limitation Pattern": c.get("limitation_pattern", ""),
            "Supporting Papers": ", ".join(c.get("supporting_papers", [])),
            "Possible Field Gap": c.get("possible_field_gap", ""),
            "Confidence": c.get("confidence", ""),
        })

    tradeoff_rows = []
    for t in reasoning.get("tradeoffs", []):
        tradeoff_rows.append({
            "Tradeoff ID": t.get("tradeoff_id", ""),
            "Tradeoff Label": t.get("tradeoff_label", ""),
            "Supporting Papers": ", ".join(t.get("supporting_papers", [])),
            "Latent Gap": t.get("latent_gap", ""),
            "Confidence": t.get("confidence", ""),
        })

    gap_rows = []
    for g in reasoning.get("validated_research_gaps", []):
        gap_rows.append({
            "Research Gap": g.get("gap", ""),
            "Why It Matters": g.get("why_it_matters", ""),
            "Supporting Papers": ", ".join(g.get("supporting_papers", [])),
            "Supporting Tradeoffs": ", ".join(g.get("supporting_tradeoffs", [])),
            "Confidence": g.get("confidence", ""),
        })

    idea_rows = []
    for g in reasoning.get("validated_research_gaps", []):
        idea_rows.append({
            "Gap": g.get("gap", ""),
            "Proposed Idea": g.get("possible_research_idea", ""),
            "Required Data": g.get("required_data", ""),
            "Feasibility": g.get("feasibility", ""),
            "Risks": g.get("risks", ""),
            "Evidence": ", ".join(g.get("supporting_papers", [])),
        })

    gap_md = f"""# Gap Analysis

## Method

This gap analysis was generated using a hybrid reasoning engine:

1. Extract author-stated limitations from existing evidence.
2. Embed atomic limitation statements.
3. Cluster semantically similar limitations.
4. Merge conceptually related clusters using an LLM.
5. Build a contradiction/tradeoff graph.
6. Infer validated research gaps and research ideas.

No new papers are introduced by this tool. It reasons over existing artifacts.

---

## Merged Limitation Clusters

{_markdown_table(cluster_rows, [
    "Cluster ID",
    "Cluster Label",
    "Limitation Pattern",
    "Supporting Papers",
    "Possible Field Gap",
    "Confidence",
])}

---

## Tradeoff Graph

{_markdown_table(tradeoff_rows, [
    "Tradeoff ID",
    "Tradeoff Label",
    "Supporting Papers",
    "Latent Gap",
    "Confidence",
])}

---

## Validated Research Gaps

{_markdown_table(gap_rows, [
    "Research Gap",
    "Why It Matters",
    "Supporting Papers",
    "Supporting Tradeoffs",
    "Confidence",
])}
"""

    idea_md = f"""# Idea Bank

Research ideas are generated only from validated research gaps and tradeoffs.

{_markdown_table(idea_rows, [
    "Gap",
    "Proposed Idea",
    "Required Data",
    "Feasibility",
    "Risks",
    "Evidence",
])}
"""

    _write_file("gap_analysis.md", gap_md)
    _write_file("idea_bank.md", idea_md)


@tool
def gap_reasoning_tool(
    evidence_matrix_path: str = "evidence_matrix.md",
    source_notes_path: str = "research-notes/source_notes.md",
    raw_evidence_path: str = "raw_evidence.json",
    distance_threshold: float = 0.38,
) -> dict:
    """
    Hybrid gap reasoning tool.

    Use this after evidence_matrix.md and source notes exist.

    Pipeline:
    1. Extract author-stated limitations.
    2. Cluster limitations using embeddings.
    3. Merge conceptually related clusters with LLM.
    4. Build a tradeoff/contradiction graph.
    5. Infer validated research gaps and ideas.
    6. Write gap_analysis.md, idea_bank.md, gap_clusters.json.

    This tool does not retrieve new papers.
    """
    evidence_matrix = _read_file(evidence_matrix_path)
    source_notes = _read_file(source_notes_path)
    raw_evidence = _read_file(raw_evidence_path)

    if not evidence_matrix and not source_notes and not raw_evidence:
        return {
            "ok": False,
            "error": "No evidence_matrix.md, source_notes.md, or raw_evidence.json found. Build evidence artifacts first.",
        }

    limitations = extract_author_stated_limitations(
        evidence_matrix=evidence_matrix,
        source_notes=source_notes,
        raw_evidence=raw_evidence,
    )

    if not limitations:
        return {
            "ok": False,
            "error": "No author-stated limitations could be extracted from existing artifacts.",
        }

    raw_clusters = cluster_limitations(
        limitations=limitations,
        distance_threshold=distance_threshold,
    )

    merge_result = merge_clusters_conceptually(raw_clusters)
    merged_clusters = merge_result.get("merged_clusters", [])

    tradeoff_graph = build_tradeoff_graph(merged_clusters)

    reasoning = infer_gaps_from_clusters_and_tradeoffs(
        merged_clusters=merged_clusters,
        tradeoff_graph=tradeoff_graph,
    )

    write_gap_outputs(
        limitations=limitations,
        raw_clusters=raw_clusters,
        merged_clusters=merged_clusters,
        tradeoff_graph=tradeoff_graph,
        reasoning=reasoning,
    )

    return {
        "ok": True,
        "num_limitations": len(limitations),
        "num_raw_clusters": len(raw_clusters),
        "num_merged_clusters": len(merged_clusters),
        "num_tradeoffs": len(tradeoff_graph.get("tradeoffs", [])),
        "num_validated_gaps": len(reasoning.get("validated_research_gaps", [])),
        "outputs": [
            "gap_clusters.json",
            "gap_analysis.md",
            "idea_bank.md",
        ],
        "summary": reasoning,
    }