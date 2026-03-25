from typing import Any
from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


_SYNTHESIS_OUTPUT_SCHEMA = """
REQUIRED OUTPUT — Return a single JSON object, no markdown:
{
  "main_sheet_exists": true/false,
  "main_sheet_name": "<exact workbook tab name or null>",
  "header_sheets": ["<exact workbook tab name>", "..."],
  "is_card_sheet": "<exact workbook TB tab name or null>",
  "technical_main_sheet": "<exact workbook tab name or null>",
  "presentation_main_sheet": "<exact workbook tab name or null>",
  "business_main_sheet": "<exact workbook tab name or null>",
  "technical_tb_sheet": "<exact workbook TB tab name or null>",
  "decision_mode": "technical_default|business_override|business_override_with_tb_validation|no_valid_sheet",
  "confidence": 0.0,
  "reasoning": "<one concise English sentence>",
  "nn_synthesis": {
    "fs_pattern_confirmed": true/false,
    "all_gates_passed": true/false,
    "layer3_role": "FS|TB|INTERMEDIATE|STAGING|UNKNOWN",
    "inter_agent_signal_agreement": true/false,
    "softmax_winner": "<exact workbook tab name or null>",
    "softmax_distribution": {"<exact workbook tab name>": 0.0},
    "tb_softmax_winner": "<exact workbook tab name or null>",
    "tb_candidate_confirmed": true/false,
    "path_to_tb_confirmed": true/false
  },
  "business_arbitration": {
    "technical_winner_sheet_type": "REPORTING_FS|ADJUSTMENT_STAGING|SOURCE_TB|INTERMEDIATE_CONSOLIDATION|AUXILIARY_SCHEDULE|UNKNOWN|null",
    "presentation_candidate": "<exact workbook tab name or null>",
    "presentation_candidate_sheet_type": "REPORTING_FS|ADJUSTMENT_STAGING|SOURCE_TB|INTERMEDIATE_CONSOLIDATION|AUXILIARY_SCHEDULE|UNKNOWN|null",
    "presentation_candidate_blocked_by": "<gate or null>",
    "presentation_candidate_disqualification_class": "CRITICAL|TECHNICAL|NONE|null",
    "override_applied": true/false
  },
  "relationship": {
    "main_to_tb_path": ["<exact workbook tab name>", "...", "<exact workbook tab name>"],
    "path_valid": false
  },
  "blocked_sheets": {
    "hidden": [],
    "tb": [],
    "no_company": [],
    "incoming_only": [],
    "staging": []
  },
  "runner_up": "<exact workbook tab name or null>",
  "suggested_manual_review": [],
  "api_response": {
    "main_sheet_exists": true/false,
    "main_sheet_name": "<exact workbook tab name or null>"
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
  4. the real header-sheet set when reporting is split across multiple tabs
  5. the TB/card sheet
  6. the validated relationship between main sheet and TB sheet

You must follow the Neural PROMAT layers exactly.
You must not invent evidence.
You must not trust sheet names alone.
You must not trust highlighted titles alone.

NON-NEGOTIABLE SHEET-IDENTITY RULE
──────────────────────────────────
A semantic reporting role is NOT the same thing as a worksheet identity.

You may reason about:
  • reporting output
  • business presentation
  • final FS role
  • external reporting structure
  • consolidated reporting layer

But every sheet-bearing field in the final JSON must be an EXACT workbook tab name.
This applies to:
  • main_sheet_name
  • header_sheets
  • technical_main_sheet
  • presentation_main_sheet
  • business_main_sheet
  • technical_tb_sheet
  • is_card_sheet
  • runner_up
  • nn_synthesis.softmax_winner
  • nn_synthesis.tb_softmax_winner
  • all softmax_distribution keys
  • business_arbitration.presentation_candidate
  • relationship.main_to_tb_path

Titles, captions, semantic labels, and inferred business entities may help classify a sheet,
but they may NEVER replace the real workbook tab name.

Examples:
  • If the visible title says "CONSOLIDATED STATEMENTS OF OPERATIONS" but the tab is `P&L`,
    you must return `P&L`.
  • If `BS` and `P&L` are both real reporting tabs, do not invent
    "External Reporting Scheme" or any similar umbrella node unless that exact
    text is a real workbook tab name.
  • If a concept cannot be mapped to an exact workbook tab name, return null.

MULTI-HEADER REPORTING RULE
───────────────────────────
Some workbooks have one real final reporting tab, for example:
  • `FS`

Other workbooks have multiple real visible final header tabs, for example:
  • `BS`
  • `P&L`
  • sometimes also `CF`

When the workbook's real business-facing reporting output is split across multiple
real worksheet tabs:
  • preserve those real tabs in `header_sheets`
  • do not collapse them into an invented parent node
  • do not replace them with a semantic umbrella label
  • choose `main_sheet_name` as the strongest single final reporting tab for API compatibility
  • but preserve the multi-sheet business truth in `header_sheets`

Therefore:
  • `main_sheet_name` is still singular
  • `header_sheets` preserves the real visible header-sheet set
  • if the correct business answer is `BS` and `P&L`, then `header_sheets` must contain both
  • never convert `BS` + `P&L` into "External Reporting Scheme"

SYNTHESIS RULES
───────────────

[NS0 — Schema recovery]
  If task_results do not contain canonical "sheet_evidence" but do contain
  per-sheet evidence under keys such as "evidence", use that structured
  per-sheet evidence as the primary NN aggregation source.
  Do not discard structured evidence merely because the envelope schema
  is imperfect.

[NS0b — Sheet-name fidelity recovery]
  Recover candidate sheets only from exact workbook tab names present in the evidence universe.
  If a candidate/evidence key/path node is not an exact workbook tab name, discard it.
  Never upgrade a semantic label into a candidate sheet.

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

[NS2b — Header-sheet preservation]
  If real visible statement-family tabs such as `FS`, `BS`, `P&L`, or `CF`
  are supported by workbook evidence as final business-facing reporting tabs,
  preserve them in `header_sheets` even if only one of them becomes the single
  `main_sheet_name`.

  Important:
    • `header_sheets` may contain one tab or multiple tabs
    • `header_sheets` must contain only exact workbook tab names
    • do not exclude `BS` or `P&L` from `header_sheets` only because they also
      have mixed structural behavior
    • if a real visible statement-family tab is a genuine business-facing header tab,
      preserve it as a header tab even if the graph also treats it as intermediate

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
  Never return a detector/title/business label unless it exactly matches a workbook tab name.

[NS7 — API contract]
  api_response must contain ONLY:
    • main_sheet_exists
    • main_sheet_name
  main_sheet_name in api_response must equal the final post-L6 / post-L7
  main reporting sheet, using an exact workbook tab name only.

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
      5. presentation candidate is an exact workbook tab name

  Step D — If override applies:
    technical_main_sheet = technical winner
    presentation_main_sheet = presentation candidate
    business_main_sheet = presentation_main_sheet
    main_sheet_name = presentation_main_sheet
    decision_mode = "business_override"

  Step E — Otherwise:
    technical_main_sheet = technical winner
    presentation_main_sheet = technical winner or best presentation-safe candidate
    business_main_sheet = presentation_main_sheet
    main_sheet_name = technical winner
    decision_mode = "technical_default"

  Step F — No synthetic override
    Never override to:
      • semantic umbrella labels
      • title-derived labels that are not real tabs
      • invented reporting entities
      • nonexistent sheet names

  Step G — Header-sheet preservation after arbitration
    After deciding the single `main_sheet_name`, preserve the real reporting
    header tabs in `header_sheets`.

    If the workbook has:
      • one real final reporting tab → `header_sheets` may contain one item
      • multiple real visible final reporting tabs such as `BS` and `P&L` →
        preserve both in `header_sheets`

    `header_sheets` must:
      • contain only exact workbook tab names
      • never contain invented umbrella labels
      • never replace real tabs with semantic report labels

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
    7. Every path node must be an exact workbook tab name

  relationship.path_valid = true only if the path is structurally supported by
  Layer-3 graph evidence and every path node is an exact workbook tab name.

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
    • the workbook clearly has multiple real header tabs and the single-winner choice is technically forced by the API contract

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
2. Discard any candidate/evidence key/path node that is not an exact workbook tab name.
3. Aggregate Layer-1 through Layer-4 evidence across agents.
4. Enforce hard gates exactly.
5. Compute the technical main-sheet winner using true-softmax-style synthesis.
6. Apply Layer-6 business arbitration.
7. Preserve the real reporting header-sheet set in `header_sheets`.
8. Apply Layer-7 TB/card-sheet validation.
9. Populate:
     • nn_synthesis
     • business_arbitration
     • relationship
     • blocked_sheets
10. Ensure main_sheet_name is the final authoritative post-arbitration main sheet.
11. Ensure `header_sheets` preserves the real visible reporting header tabs.
12. Ensure is_card_sheet is the final TB/card-sheet name or null.
13. Return JSON only.

{_SYNTHESIS_OUTPUT_SCHEMA}
""".strip()