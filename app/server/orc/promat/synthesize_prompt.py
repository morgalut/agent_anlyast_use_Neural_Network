from typing import Any
from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


_SYNTHESIS_OUTPUT_SCHEMA = """
REQUIRED OUTPUT — Return a single JSON object, no markdown:
{
  "main_sheet_exists": true/false,
  "main_sheet_name": "<sheet name or null>",
  "confidence": 0.0,
  "reasoning": "<one concise English sentence>",
  "nn_synthesis": {
    "fs_pattern_confirmed": true/false,
    "all_gates_passed": true/false,
    "layer3_role": "FS|TB|INTERMEDIATE|UNKNOWN",
    "inter_agent_signal_agreement": true/false,
    "softmax_winner": "<sheet name>",
    "softmax_distribution": {"<sheet>": 0.0}
  },
  "blocked_sheets": {
    "hidden": [], "tb": [], "no_company": [], "incoming_only": []
  },
  "runner_up": "<sheet name or null>",
  "suggested_manual_review": [],
  "api_response": {
    "main_sheet_exists": true/false,
    "main_sheet_name": "<sheet name or null>"
  }
}
"""


def build_synthesize_prompt(
    analysis: str,
    plan: str,
    main_sheet_result: Any,
    task_results: Any,
) -> str:
    """
    ORC Synthesis-node entry point.

    Replaces S1–S5 scoring rules with NN-layer aggregation across agents.
    This node owns the final authoritative decision.
    """
    return f"""
{FULL_NN_PROMAT}

╔══════════════════════════════════════════════════════════╗
║  AGENT ROLE: ORC Synthesis Node (Final Authority)       ║
╚══════════════════════════════════════════════════════════╝

You are the Synthesis node — the final decision gate in the ORC pipeline.
You aggregate NN-layer evidence from all prior agents and issue the
authoritative verdict on the main sheet.

SYNTHESIS RULES
───────────────

[NS1 — Cross-agent signal aggregation]
  For each candidate sheet, collect nn_evidence from ALL task_results.
  A signal that fires (= 1) in ≥ 2 agents → CONFIRMED signal.
  A signal that fires in exactly 1 agent → WEAK signal.
  Use only CONFIRMED signals when computing the final softmax.

[NS2 — FS_PATTERN aggregation]
  If FS_PATTERN = 1 in ≥ 1 agent AND Layer-4 passed → strong candidate.
  If FS_PATTERN = 1 in one agent but PARTIAL_FS_PATTERN in another:
    Apply Layer-3 A1 graph rule: the sheet that points to a TB = FS winner.

[NS3 — Hard gate enforcement (NON-NEGOTIABLE)]
  Every proposed sheet MUST have layer4.passed = true.
  A sheet with blocked_by ≠ null → PERMANENTLY DISQUALIFIED.
  No level of inter-agent agreement can override a fired gate.
  This rule supersedes NS1 and NS2.

[NS4 — Softmax aggregation across agents]
  For each candidate sheet:
    avg_S(sheet) = mean of S(sheet) values reported by all agents
  Re-normalise using softmax:
    p(sheet) = avg_S(sheet) / Σ avg_S(all passing sheets)
  Select the sheet with highest p.

[NS5 — Confidence thresholds]
  p ≥ 0.70 → CONFIRMED  (main_sheet_exists = true, confident)
  p ≥ 0.40 → POSSIBLE   (main_sheet_exists = true, uncertain)
  p < 0.40 → NOT FOUND  (main_sheet_exists = false)

[NS6 — Detector override rule]
  The heuristic detector (main_sheet_result) is a HINT only.
  If the NN softmax winner differs from the detector hint → winner wins.
  Never revert to the detector result when agents disagree.

[NS7 — API contract]
  api_response must contain ONLY main_sheet_exists and main_sheet_name.
  All evidence and reasoning stays inside nn_synthesis.

LIVE INPUT
──────────
Analysis node output (NN layer evidence):
{analysis}

Plan node output:
{plan}

Detector hint (not authoritative):
{main_sheet_result}

Execution agent results (primary evidence source):
{task_results}

INSTRUCTIONS
────────────
1. Collect nn_evidence from every entry in task_results (NS1).
2. For each sheet: check Layer-4 gate status (NS3) — discard if blocked.
3. Compute avg_S per passing sheet and re-normalise softmax (NS4).
4. Apply confidence thresholds (NS5).
5. Return the output JSON.

{_SYNTHESIS_OUTPUT_SCHEMA}
""".strip()