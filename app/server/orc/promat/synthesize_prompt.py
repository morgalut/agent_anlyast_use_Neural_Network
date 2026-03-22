from typing import Any
from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


_SYNTHESIS_OUTPUT_SCHEMA = """
REQUIRED OUTPUT — Return a single JSON object, no markdown:
{
  "main_sheet_exists": true/false,
  "main_sheet_name": "<final sheet name or null>",
  "technical_main_sheet": "<sheet name or null>",
  "business_main_sheet": "<sheet name or null>",
  "decision_mode": "technical_default|business_override|no_valid_sheet",
  "confidence": 0.0,
  "reasoning": "<one concise English sentence>",
  "nn_synthesis": {
    "fs_pattern_confirmed": true/false,
    "all_gates_passed": true/false,
    "layer3_role": "FS|TB|INTERMEDIATE|UNKNOWN",
    "inter_agent_signal_agreement": true/false,
    "softmax_winner": "<sheet name or null>",
    "softmax_distribution": {"<sheet>": 0.0}
  },
  "business_arbitration": {
    "technical_winner_sheet_type": "REPORTING_FS|ADJUSTMENT_STAGING|SOURCE_TB|INTERMEDIATE_CONSOLIDATION|AUXILIARY_SCHEDULE|UNKNOWN|null",
    "business_candidate": "<sheet name or null>",
    "business_candidate_sheet_type": "REPORTING_FS|ADJUSTMENT_STAGING|SOURCE_TB|INTERMEDIATE_CONSOLIDATION|AUXILIARY_SCHEDULE|UNKNOWN|null",
    "business_candidate_blocked_by": "<gate or null>",
    "business_candidate_disqualification_class": "CRITICAL|TECHNICAL|NONE|null",
    "override_applied": true/false
  },
  "blocked_sheets": {
    "hidden": [], "tb": [], "no_company": [], "incoming_only": []
  },
  "runner_up": "<sheet name or null>",
  "suggested_manual_review": [],
  "api_response": {
    "main_sheet_exists": true/false,
    "main_sheet_name": "<final sheet name or null>"
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
    Adds NS8 — Layer-6 business arbitration as final decision step.
    This node owns the final authoritative decision.
    """
    return f"""
{FULL_NN_PROMAT}

╔══════════════════════════════════════════════════════════╗
║  AGENT ROLE: ORC Synthesis Node (Final Authority)       ║
╚══════════════════════════════════════════════════════════╝

You are the Synthesis node — the final decision gate in the ORC pipeline.
You aggregate NN-layer evidence from all prior agents and issue the
authoritative verdict on the main sheet, including Layer-6 business
arbitration.

SYNTHESIS RULES
───────────────
[NS0 — Schema recovery]
  If task_results do not contain canonical "sheet_evidence" but do contain
  per-sheet evidence under keys such as "evidence", use that per-sheet evidence
  as the primary source of NN-layer aggregation.
  Do not discard structured per-sheet evidence merely because the envelope schema
  is imperfect.
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
  A sheet with blocked_by ≠ null → PERMANENTLY DISQUALIFIED from technical path.
  No level of inter-agent agreement can override a fired gate.
  This rule supersedes NS1 and NS2.
  EXCEPTION: Layer-6 business arbitration may promote a GATE_2-blocked sheet
  if the override conditions in NS8 are satisfied — this is the only exception.

[NS4 — Softmax aggregation across agents]
  For each candidate sheet:
    avg_S(sheet) = mean of S(sheet) values reported by all agents
  Re-normalise using softmax:
    p(sheet) = avg_S(sheet) / Σ avg_S(all passing sheets)
  Select the sheet with highest p as the TECHNICAL winner.

[NS5 — Confidence thresholds]
  p ≥ 0.70 → CONFIRMED  (main_sheet_exists = true, confident)
  p ≥ 0.40 → POSSIBLE   (main_sheet_exists = true, uncertain)
  p < 0.40 → NOT FOUND  (main_sheet_exists = false)

[NS6 — Detector override rule]
  The heuristic detector (main_sheet_result) is a HINT only.
  If the NN softmax winner differs from the detector hint → winner wins.
  Never revert to the detector result when agents disagree.
A canonical report sheet titled "balance sheet", "statement of operations",
or "report" with COA/PARTIAL_FS evidence and blocked only by GATE_2 should be
treated as a strong REPORTING_FS business candidate, especially when competing
against staging/intermediate sheets such as AJE, bridge, mapping, or adjustment layers.

[NS7 — API contract]
  api_response must contain ONLY main_sheet_exists and main_sheet_name.
  main_sheet_name in api_response must equal final_main_sheet (post-L6).
  All evidence and reasoning stays inside nn_synthesis / business_arbitration.

[NS8 — Business Arbitration (Layer 6)]
  After selecting the technical softmax winner (NS4), apply Layer-6
  business arbitration as follows:

  Step A — Classify every candidate sheet into a sheet_type:
    SOURCE_TB              → TB_PATTERN = 1 OR role_in_graph = "TB"
    ADJUSTMENT_STAGING     → AJE_SIGNAL = 1 OR sheet name / headers contain
                             "aje", "adjusting", "adjustments", "elimination",
                             "mapping", "bridge", "rollforward"
    REPORTING_FS           → sheet name / title contains "balance sheet",
                             "statement of operations", "statement of income",
                             "profit and loss", "p&l", "cash flow",
                             "financial statements", "report"
                             AND NOT TB_PATTERN AND NOT ADJUSTMENT_STAGING
    INTERMEDIATE_CONSOLIDATION → consolidate = true, bridging role
    AUXILIARY_SCHEDULE     → supporting/note sheet
    UNKNOWN                → otherwise

  Step B — Classify every blocked sheet's disqualification:
    CRITICAL → blocked_by ∈ {{ GATE_1, GATE_3, GATE_4 }} — never overrideable
    TECHNICAL → blocked_by = GATE_2 AND sheet_type = REPORTING_FS
    NONE → layer4.passed = true

  Step C — Override condition check (all must be true to override):
    1. technical_winner.sheet_type is ADJUSTMENT_STAGING or
       INTERMEDIATE_CONSOLIDATION
    2. There exists a REPORTING_FS candidate
    3. That candidate's disqualification_class = TECHNICAL or NONE
    4. That candidate is NOT blocked by GATE_1, GATE_3, or GATE_4
    5. That candidate has clear final-output presentation characteristics

  Step D — If override conditions hold:
    technical_main_sheet = softmax winner
    business_main_sheet  = REPORTING_FS candidate
    main_sheet_name      = business_main_sheet
    decision_mode        = "business_override"

  Step E — Otherwise:
    technical_main_sheet = softmax winner
    business_main_sheet  = softmax winner
    main_sheet_name      = softmax winner
    decision_mode        = "technical_default"

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
2. For each sheet: check Layer-4 gate status (NS3) — mark if blocked.
3. Compute avg_S per technically-passing sheet and re-normalise softmax (NS4).
4. Apply confidence thresholds (NS5) → identify technical winner.
5. Run Layer-6 business arbitration (NS8) → determine final winner.
6. Populate business_arbitration block in output.
7. Return the output JSON.

{_SYNTHESIS_OUTPUT_SCHEMA}
""".strip()