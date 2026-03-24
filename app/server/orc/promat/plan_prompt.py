from typing import Any
from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


def build_plan_prompt(analysis: str, main_sheet_result: Any) -> str:
    """
    ORC Plan-node entry point.

    Replaces old threshold-based planning with NN-signal-driven planning.
    The plan targets unresolved structural questions across:
      • main-sheet detection
      • staging / intermediate disambiguation
      • TB/card-sheet validation
      • main-to-TB relationship validation
    """
    return f"""
{FULL_NN_PROMAT}

╔══════════════════════════════════════════════════════════╗
║  AGENT ROLE: ORC Plan Node                              ║
╚══════════════════════════════════════════════════════════╝

You are the Plan node in a LangGraph ORC pipeline.
You received workbook analysis and detector hints.
Your job is to translate unresolved Neural PROMAT questions into a focused
verification plan for the Research / Execution agent.

Your plan must follow the 7-layer Neural PROMAT exactly.

PLANNING PRINCIPLES
───────────────────
• Plan around structural uncertainty, not around titles alone.
• Do not trust sheet names alone.
• Do not trust highlighted titles alone.
• Prioritize graph direction, formula evidence, and source/output structure.
• The final system must identify:
    1. the technical main sheet
    2. the final/presentation main sheet
    3. the TB/card sheet
    4. the validated relationship between them

PLANNING RULES
──────────────

[P1 — Plan around FS_PATTERN gaps]
  If a candidate appears to be a reporting sheet:
    verify:
      • COA_SIGNAL
      • COMPANY_COLUMN_SIGNAL
      • CROSS_REF_SIGNAL
      • whether formulas point outward to supporting sheets
    confirm that the graph direction is:
      reporting → intermediate/staging → TB
    and not the reverse.

[P2 — Plan around PARTIAL_FS_PATTERN]
  If a candidate shows only partial FS evidence:
    deepen the scan:
      • inspect AJE_SIGNAL
      • inspect CONSOLIDATE_SIGNAL
      • inspect whether missing company evidence is real or only weakly labelled
      • inspect whether the sheet is presentation-final but technically incomplete
    also check if Layer-3 graph evidence promotes it.

[P3 — Plan around TB/card-sheet detection]
  For suspected TB/source sheets, verify:
    • HAS_CODE_COLUMN
    • HAS_DESCRIPTION_COLUMN
    • HAS_FINAL_COLUMN
    • FINAL_REFERENCE_SIGNAL
    • TB_REFERENCE_SIGNAL
  If several code-like columns exist:
    prefer the one with fewer repetitions and better adjacency to description.
  If several final columns exist:
    prefer the one referenced by upstream formulas or behaving as the main
    ending-balance numeric column.

[P4 — Plan around STRONG_TB_PATTERN]
  If a sheet looks like a TB candidate:
    verify whether it is a strong TB candidate by confirming:
      • source-like graph role
      • referenced-by behavior
      • valid main → ... → TB path
    This is higher priority than title-based hints such as "TB" or "GL".

[P5 — Plan for single-company-column CONSOLIDATE logic]
  If COMPANY_COLUMN_SIGNAL appears present but only one company-like amount
  column exists:
    inspect referenced intermediate sheets.
    Check whether an intermediate sheet contains:
      • COA structure
      • multiple company columns
    If yes, treat the current sheet as possible consolidated presentation output.

[P6 — Plan around staging / AJE ambiguity]
  If a sheet may be AJE / adjusting / bridge / mapping / support / rollforward:
    verify:
      • STAGING_ROLE_SIGNAL
      • STAGING_PATTERN
      • aje_source_role
      • whether the sheet primarily feeds AJE columns of another sheet
    Such a sheet may be important structurally but must not become the final
    main reporting sheet.

[P7 — Plan around graph tie-breaking]
  If two reporting candidates are close:
    compare:
      • which sheet points to the other
      • which sheet sits upstream vs downstream
      • which sheet has a valid path to TB
    Prefer the reporting-origin sheet over source/staging sheets.

[P8 — Plan around hard gates]
  Explicitly verify whether any candidate should trigger:
    • GATE_1 — hidden
    • GATE_2 — no company/consolidate evidence
    • GATE_3 — TB sheet
    • GATE_4 — pure source
    • GATE_5 — staging/AJE-support sheet
  Remember:
    • GATE_2 is TECHNICAL
    • GATE_1 / GATE_3 / GATE_4 / GATE_5 are CRITICAL

[P9 — Plan around Layer-7 relationship validation]
  For the strongest reporting candidate, explicitly verify:
    • direct path: main → TB
    • indirect path: main → intermediate → TB
    • indirect path: main → staging → TB
  Preserve the strongest valid path.
  The final result must include:
    relationship.main_to_tb_path
    relationship.path_valid

[P10 — Plan for business arbitration]
  If a canonical report-like sheet may be blocked only by GATE_2:
    verify whether it is a safe presentation candidate by checking:
      • final-output layout
      • report-style structure
      • COA support
      • not TB
      • not staging
  This supports Layer-6 business arbitration.

[P11 — API contract awareness]
  The final API response must expose only:
    • main_sheet_exists
    • main_sheet_name
  But the research plan MUST still collect enough evidence to support:
    • technical_main_sheet
    • presentation_main_sheet
    • is_card_sheet
    • technical_tb_sheet
    • relationship
    • decision_mode

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
      "rule_id": "P1|P2|P3|P4|P5|P6|P7|P8|P9|P10",
      "nn_signal": "<signal or pattern name from NN layers>",
      "description": "<what to verify in plain English>",
      "action": "scan_layer1 | scan_layer3_graph | check_intermediate_sheet | verify_formulas | verify_tb_columns | verify_staging_role | verify_path_to_tb",
      "expected_activation": "<which signal/pattern/path should fire or be confirmed>"
    }}
  ],
  "fallback_candidate": "<sheet name or null>",
  "tb_candidate": "<sheet name or null>",
  "discard_sheets": ["<sheets blocked by GATE_1 through GATE_5>"],
  "relationship_goal": {{
    "main_to_tb_path": ["<candidate main>", "...", "<candidate tb>"],
    "path_must_be_validated": true
  }},
  "api_response": {{
    "main_sheet_exists": true,
    "main_sheet_name": "<sheet name>"
  }}
}}
""".strip()