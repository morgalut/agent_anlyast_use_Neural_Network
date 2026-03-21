from typing import Any
from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


def build_plan_prompt(analysis: str, main_sheet_result: Any) -> str:
    """
    ORC Plan-node entry point.

    Replaces threshold-based planning (T1–T5) with NN-signal-driven planning.
    The plan targets specific unresolved signals, not score gaps.
    """
    return f"""
{FULL_NN_PROMAT}

╔══════════════════════════════════════════════════════════╗
║  AGENT ROLE: ORC Plan Node                              ║
╚══════════════════════════════════════════════════════════╝

You are the Plan node in a LangGraph ORC pipeline.
You received NN layer output from the Analyze node.
Your job: translate unresolved NN signals into a focused verification plan
for the Research / Execution agent.

PLANNING RULES
──────────────

[P1 — Plan around FS_PATTERN gaps]
  If the candidate lit FS_PATTERN = 1:
    Verify COMPANY_COLUMN_SIGNAL and CROSS_REF_SIGNAL with direct tool calls.
    Confirm that the Layer-3 graph shows this sheet → TB (not the reverse).

[P2 — Plan around PARTIAL_FS_PATTERN]
  If the candidate lit only PARTIAL_FS_PATTERN = 1:
    Deepen the scan: check AJE_SIGNAL, CONSOLIDATE_SIGNAL.
    Check for an intermediate sheet (Layer 3 A3).

[P3 — Plan for close-confidence tie (< 0.10 gap between top two)]
  Apply Layer-3 A1+A2: which sheet points to which?
  The pointing sheet = FS.  The pointed-to sheet = TB.

[P4 — Plan for single-company-column CONSOLIDATE check]
  If COMPANY_COLUMN_SIGNAL = 1 but only one company column found:
    Check intermediate sheet (Layer 3 A4):
    Does it contain COA structure AND multiple company columns?
    If yes → current sheet is a CONSOLIDATE, still valid main sheet.

[P5 — API contract]
  The api_response block must contain ONLY:
    main_sheet_exists (bool) + main_sheet_name (string)
  All detailed reasoning stays inside the plan JSON — never in the API response.

LIVE INPUT
──────────
NN analysis output:
{analysis}

Detector hint (for reference — not authoritative):
{main_sheet_result}

REQUIRED OUTPUT — Return a single JSON object, no markdown:
{{
  "verification_target": "<sheet name>",
  "checks": [
    {{
      "rule_id": "P1",
      "nn_signal": "<signal or pattern name from NN layers>",
      "description": "<what to verify in plain English>",
      "action": "scan_layer1 | scan_layer3_graph | check_intermediate_sheet | verify_formulas",
      "expected_activation": "<which signal should fire>"
    }}
  ],
  "fallback_candidate": "<sheet name or null>",
  "discard_sheets": ["<sheets blocked by GATE_1 through GATE_4>"],
  "api_response": {{
    "main_sheet_exists": true,
    "main_sheet_name": "<sheet name>"
  }}
}}
""".strip()