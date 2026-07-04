import os
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from langchain_core.tools import tool


PROJECT_ROOT = Path("/home/aiphuongnguyen/medical-deep-agent")
MODEL = os.getenv("ARTIFACT_BUILDER_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _safe_json_loads(response.choices[0].message.content)


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]

    for row in rows:
        values = []
        for col in columns:
            value = str(row.get(col, "Not specified")).replace("\n", " ").strip()
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


@tool
def artifact_builder_tool(
    raw_evidence_path: str = "raw_evidence.json",
    source_notes_path: str = "research-notes/source_notes.md",
    evidence_matrix_path: str = "evidence_matrix.md",
    citation_registry_path: str = "citation_registry.md",
) -> dict:
    """
    Build structured research artifacts from retrieved paper evidence.

    Use this AFTER academic search tools have produced or saved raw evidence.
    This tool creates:
    - research-notes/source_notes.md
    - evidence_matrix.md
    - citation_registry.md

    It does not search for new papers.
    It only structures existing evidence.
    """
    raw_text = _read_file(raw_evidence_path)

    if not raw_text:
        return {
            "ok": False,
            "error": f"No raw evidence found at {raw_evidence_path}. Save retrieved arXiv/PubMed/Web results there first.",
        }

    system_prompt = """
You are an academic evidence artifact builder.

Your job is to convert retrieved paper/source evidence into:
1. source notes
2. evidence matrix rows
3. citation registry rows

Rules:
- Do not invent papers, URLs, metrics, datasets, or limitations.
- Use only information present in the raw evidence.
- If a field is missing, write "Not specified".
- Preserve author-stated limitations only.
- Do not replace limitations with generic medical AI limitations.
- Return valid JSON only.
"""

    user_prompt = f"""
Convert the raw evidence below into structured artifacts.

Return JSON with this schema:

{{
  "source_notes": [
    {{
      "source_id": "S1",
      "title": "...",
      "year": "...",
      "source_tool": "arxiv_search|pubmed_search|web_search|unknown",
      "source_type": "preprint|biomedical_literature|official|web|unknown",
      "url": "...",
      "authors": "...",
      "problem": "...",
      "method": "...",
      "dataset": "...",
      "supervision_setting": "...",
      "metrics": "...",
      "validation_setting": "...",
      "main_findings": "...",
      "author_stated_limitations": "...",
      "uncertainty": "..."
    }}
  ],
  "evidence_matrix": [
    {{
      "Paper": "...",
      "Year": "...",
      "Method": "...",
      "Dataset": "...",
      "Supervision Setting": "...",
      "Metrics": "...",
      "Validation Setting": "...",
      "Limitations": "...",
      "Source Type": "...",
      "Source": "S1"
    }}
  ],
  "citation_registry": [
    {{
      "Source ID": "S1",
      "Title": "...",
      "Year": "...",
      "Source Tool": "...",
      "Source Type": "...",
      "URL": "...",
      "Notes": "..."
    }}
  ]
}}

Raw evidence:
=============
{raw_text}
"""

    data = _llm_json(system_prompt, user_prompt)

    source_notes = data.get("source_notes", [])
    evidence_rows = data.get("evidence_matrix", [])
    registry_rows = data.get("citation_registry", [])

    notes_md = "# Source Notes\n\n"
    for note in source_notes:
        notes_md += f"""## {note.get("source_id", "S?")} — {note.get("title", "Not specified")}

Title: {note.get("title", "Not specified")}
Year: {note.get("year", "Not specified")}
Source Tool: {note.get("source_tool", "Not specified")}
Source Type: {note.get("source_type", "Not specified")}
URL: {note.get("url", "Not specified")}
Authors: {note.get("authors", "Not specified")}

Problem:
{note.get("problem", "Not specified")}

Method:
{note.get("method", "Not specified")}

Dataset:
{note.get("dataset", "Not specified")}

Supervision Setting:
{note.get("supervision_setting", "Not specified")}

Metrics:
{note.get("metrics", "Not specified")}

Validation Setting:
{note.get("validation_setting", "Not specified")}

Main Findings:
{note.get("main_findings", "Not specified")}

Author-Stated Limitations:
{note.get("author_stated_limitations", "Not specified")}

Uncertainty:
{note.get("uncertainty", "Not specified")}

---

"""

    matrix_md = "# Evidence Matrix\n\n"
    matrix_md += _markdown_table(
        evidence_rows,
        [
            "Paper",
            "Year",
            "Method",
            "Dataset",
            "Supervision Setting",
            "Metrics",
            "Validation Setting",
            "Limitations",
            "Source Type",
            "Source",
        ],
    )

    registry_md = "# Citation Registry\n\n"
    registry_md += _markdown_table(
        registry_rows,
        [
            "Source ID",
            "Title",
            "Year",
            "Source Tool",
            "Source Type",
            "URL",
            "Notes",
        ],
    )

    _write_file(source_notes_path, notes_md)
    _write_file(evidence_matrix_path, matrix_md)
    _write_file(citation_registry_path, registry_md)

    return {
        "ok": True,
        "outputs": [
            source_notes_path,
            evidence_matrix_path,
            citation_registry_path,
        ],
        "num_sources": len(source_notes),
        "num_matrix_rows": len(evidence_rows),
        "num_registry_rows": len(registry_rows),
    }