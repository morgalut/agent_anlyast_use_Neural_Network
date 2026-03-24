from typing import Any
from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


_SYNTHESIS_OUTPUT_SCHEMA = """
REQUIRED OUTPUT — Return a single JSON object, no markdown:
{
  "main_sheet_exists": true/false,
  "main_sheet_name": "<final sheet name or null>",
  "is_card_sheet": "<TB sheet name or null>",
  "technical_main_sheet": "<sheet name or null>",
  "presentation_main_sheet": "<sheet name or null>",
  "technical_tb_sheet": "<TB sheet name or null>",
  "decision_mode": "technical_default|business_override|business_override_with_tb_validation|no_valid_sheet",
  "confidence": 0.0,
  "reasoning": "<one concise English sentence>",
  "nn_synthesis": {
    "fs_pattern_confirmed": true/false,
    "all_gates_passed": true/false,
    "layer3_role": "FS|TB|INTERMEDIATE|STAGING|UNKNOWN",
    "inter_agent_signal_agreement": true/false,
    "softmax_winner": "<sheet name or null>",
    "softmax_distribution": {"<sheet>": 0.0},
    "tb_softmax_winner": "<sheet name or null>",
    "tb_candidate_confirmed": true/false,
    "path_to_tb_confirmed": true/false
  },
  "business_arbitration": {
    "technical_winner_sheet_type": "REPORTING_FS|ADJUSTMENT_STAGING|SOURCE_TB|INTERMEDIATE_CONSOLIDATION|AUXILIARY_SCHEDULE|UNKNOWN|null",
    "presentation_candidate": "<sheet name or null>",
    "presentation_candidate_sheet_type": "REPORTING_FS|ADJUSTMENT_STAGING|SOURCE_TB|INTERMEDIATE_CONSOLIDATION|AUXILIARY_SCHEDULE|UNKNOWN|null",
    "presentation_candidate_blocked_by": "<gate or null>",
    "presentation_candidate_disqualification_class": "CRITICAL|TECHNICAL|NONE|null",
    "override_applied": true/false
  },
  "relationship": {
    "main_to_tb_path": [],
    "path_valid": false
  },
  "blocked_sheets": {
    "hidden": [],
    "tb": [],
    "no_company": [],
    "incoming_only": [],
    "staging": []
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

    This node is the final authority.
    It aggregates NN-layer evidence across prior agents, determines the technical
    main-sheet winner, applies Layer-6 business arbitration, and then applies
    Layer-7 TB/card-sheet validation.
    """
    return f"""
{FULL_NN_PROMAT}

╔══════════════════════════════════════════════════════════╗
║  AGENT ROLE: ORC Synthesis Node (Final Authority)       ║
╚══════════════════════════════════════════════════════════╝

You are the Synthesis node — the final decision gate in the ORC pipeline.
You aggregate NN-layer evidence from all prior agents and issue the
authoritative verdict on:

  1. the final main reporting sheet
  2. the technical main sheet
  3. the presentation/business main sheet
  4. the TB/card sheet
  5. the validated relationship between main sheet and TB sheet

You must follow the Neural PROMAT layers exactly.
You must not invent evidence.
You must not trust sheet names alone.
You must not trust highlighted titles alone.

SYNTHESIS RULES
───────────────

[NS0 — Schema recovery]
  If task_results do not contain canonical "sheet_evidence" but do contain
  per-sheet evidence under keys such as "evidence", use that structured
  per-sheet evidence as the primary NN aggregation source.
  Do not discard structured evidence merely because the envelope schema
  is imperfect.

[NS1 — Cross-agent evidence aggregation]
  For each candidate sheet, collect NN-layer evidence from ALL task_results.
  A signal that fires (= 1) in ≥ 2 agents → CONFIRMED signal.
  A signal that fires in exactly 1 agent → WEAK signal.
  Prefer CONFIRMED signals when synthesizing final technical conclusions.

[NS2 — FS-pattern aggregation]
  If FS_PATTERN = 1 in ≥ 1 agent AND Layer-4 passed, treat as strong technical candidate.
  If FS_PATTERN = 1 in one agent but PARTIAL_FS_PATTERN in another:
    apply graph direction and Layer-3 path evidence.
  Prefer the sheet that behaves as:
    reporting output → intermediate/staging → TB
  over the sheet that behaves as a source or staging layer.

[NS3 — Hard gate enforcement (NON-NEGOTIABLE)]
  Every proposed TECHNICAL main sheet must have layer4.passed = true.
  A sheet with blocked_by ≠ null is disqualified from the technical path.
  No amount of inter-agent agreement may override a fired technical gate.
  EXCEPTION:
    Layer-6 business arbitration may later promote a GATE_2-blocked sheet
    if and only if it is a safe REPORTING_FS presentation candidate.
  CRITICAL blocks must never be overridden:
    GATE_1, GATE_3, GATE_4, GATE_5

[NS4 — True softmax aggregation across agents]
  For each technically passing candidate sheet:
    aggregate technical support into a main-sheet score consistent with Layer 5.
  Use TRUE SOFTMAX logic, not simple linear ratio normalisation.

  True softmax principle:
    p(sheet_i) = exp(z_i / τ) / Σ exp(z_j / τ)

  Required synthesis behavior:
    • a strong FS candidate must dominate weaker partial candidates
    • noisy partial sheets must not dilute the winner
    • do NOT use simple p = S / ΣS phrasing as the governing math

  Select the highest-probability passing sheet as the TECHNICAL main-sheet winner.

[NS5 — Confidence thresholds]
  confidence ≥ 0.70 → CONFIRMED
  confidence ≥ 0.40 → POSSIBLE
  confidence < 0.40 → NOT FOUND

  Therefore:
    • confidence ≥ 0.70 → main_sheet_exists = true, strong confidence
    • confidence ≥ 0.40 → main_sheet_exists = true, uncertain
    • confidence < 0.40 → main_sheet_exists = false

[NS6 — Detector rule]
  The heuristic detector in main_sheet_result is a hint only.
  If the NN synthesis winner differs from the detector hint, the NN synthesis winner wins.
  Never revert to the detector result merely because its title sounds right.

[NS7 — API contract]
  api_response must contain ONLY:
    • main_sheet_exists
    • main_sheet_name
  main_sheet_name in api_response must equal the final post-L6 / post-L7
  main reporting sheet.

[NS8 — Layer-6 business arbitration]
  After selecting the technical main-sheet winner, apply Layer 6.

  Step A — Classify every candidate sheet into a sheet_type:
    SOURCE_TB
      → TB_PATTERN = 1
      OR STRONG_TB_PATTERN = 1
      OR role_in_graph = "TB"

    ADJUSTMENT_STAGING
      → STAGING_PATTERN = 1
      OR AJE_SIGNAL = 1 with staging/support behavior
      OR aje_source_role = true
      OR role_in_graph = "STAGING"

    REPORTING_FS
      → FS_PATTERN = 1 and not staging
      OR strong final-output presentation evidence with COA support

    INTERMEDIATE_CONSOLIDATION
      → consolidate = true
      OR sheet bridges reporting and source but is not final presentation output

    AUXILIARY_SCHEDULE
      → supporting/note/schedule-like sheet, not final output

    UNKNOWN
      → otherwise

  Step B — Classify blocked sheets:
    CRITICAL
      → blocked_by ∈ {{ GATE_1, GATE_3, GATE_4, GATE_5 }}

    TECHNICAL
      → blocked_by = GATE_2 and sheet_type = REPORTING_FS

    NONE
      → layer4.passed = true

  Step C — Safe override conditions
    Override only if ALL are true:
      1. technical_winner is not the clearest human-facing final report
      2. presentation candidate is REPORTING_FS
      3. presentation candidate is not critically blocked
      4. presentation candidate shows clear FINAL_OUTPUT_ROLE_SIGNAL behavior

  Step D — If override applies:
    technical_main_sheet = technical winner
    presentation_main_sheet = presentation candidate
    main_sheet_name = presentation_main_sheet
    decision_mode = "business_override"

  Step E — Otherwise:
    technical_main_sheet = technical winner
    presentation_main_sheet = technical winner or best presentation-safe candidate
    main_sheet_name = technical winner
    decision_mode = "technical_default"

[NS9 — Layer-7 TB/card-sheet validation]
  After final main-sheet determination, run Layer 7.

  The TB/card sheet is a FIRST-CLASS final output.
  You must determine:
    • is_card_sheet
    • technical_tb_sheet
    • relationship.main_to_tb_path
    • relationship.path_valid

  TB selection rules:
    1. Prefer STRONG_TB_PATTERN over TB_PATTERN
    2. Prefer candidates with:
         HAS_CODE_COLUMN = 1
         HAS_DESCRIPTION_COLUMN = 1
         HAS_FINAL_COLUMN = 1
    3. Prefer FINAL_REFERENCE_SIGNAL / TB_REFERENCE_SIGNAL / REFERENCED_BY_SIGNAL
    4. Prefer candidates reachable from the final main sheet by:
         main → TB
         main → intermediate → TB
         main → staging → TB
    5. Prefer visible evidence-backed TB sheets
    6. Do not choose TB by sheet name alone

  relationship.path_valid = true only if the path is structurally supported by
  Layer-3 graph evidence.

[NS10 — Decision-mode upgrade with TB validation]
  If:
    • decision_mode = "business_override"
    • is_card_sheet is not null
    • relationship.path_valid = true

  then upgrade:
    decision_mode = "business_override_with_tb_validation"

[NS11 — Blocked-sheet reporting]
  Populate blocked_sheets with categories:
    • hidden
    • tb
    • no_company
    • incoming_only
    • staging

[NS12 — Manual review discipline]
  Suggest manual review only when evidence is genuinely ambiguous, such as:
    • two strong reporting candidates remain close
    • final main sheet is valid but TB relationship is weak
    • only hidden TB candidates exist
    • path_to_tb is unresolved

LIVE INPUT
──────────
Analysis node output:
{analysis}

Plan node output:
{plan}

Detector hint (not authoritative):
{main_sheet_result}

Execution agent results (primary evidence source):
{task_results}

INSTRUCTIONS
────────────
1. Recover all structured per-sheet NN evidence from task_results.
2. Aggregate Layer-1 through Layer-4 evidence across agents.
3. Enforce hard gates exactly.
4. Compute the technical main-sheet winner using true-softmax-style synthesis.
5. Apply Layer-6 business arbitration.
6. Apply Layer-7 TB/card-sheet validation.
7. Populate:
     • nn_synthesis
     • business_arbitration
     • relationship
     • blocked_sheets
8. Ensure main_sheet_name is the final authoritative post-arbitration main sheet.
9. Ensure is_card_sheet is the final TB/card-sheet name or null.
10. Return JSON only.

{_SYNTHESIS_OUTPUT_SCHEMA}
""".strip()