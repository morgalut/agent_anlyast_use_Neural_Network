from typing import Any
from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


_TASK_OUTPUT_SCHEMA = """
REQUIRED OUTPUT — Return a single JSON object, no markdown:
{
  "main_sheet_exists": true/false,
  "main_sheet_name": "<sheet name or null>",
  "is_card_sheet": "<TB sheet name or null>",
  "technical_main_sheet": "<sheet name or null>",
  "presentation_main_sheet": "<sheet name or null>",
  "technical_tb_sheet": "<TB sheet name or null>",
  "main_sheet_confirmed": true/false,
  "confidence": 0.0,
  "decision_mode": "technical_default|business_override|business_override_with_tb_validation|no_valid_sheet",
  "sheet_evidence": {
    "<sheet name>": {
      "layer1": {
        "COA_SIGNAL": 0,
        "FORMULA_SIGNAL": 0,
        "CROSS_REF_SIGNAL": 0,
        "REFERENCED_BY_SIGNAL": 0,
        "COMPANY_COLUMN_SIGNAL": 0,
        "AJE_SIGNAL": 0,
        "CONSOLIDATE_SIGNAL": 0,
        "HAS_CODE_COLUMN": 0,
        "HAS_DESCRIPTION_COLUMN": 0,
        "HAS_FINAL_COLUMN": 0,
        "FINAL_REFERENCE_SIGNAL": 0,
        "TB_REFERENCE_SIGNAL": 0,
        "STAGING_ROLE_SIGNAL": 0,
        "HIDDEN_SIGNAL": 0
      },
      "layer2": {
        "FS_PATTERN": 0,
        "TB_PATTERN": 0,
        "PARTIAL_FS_PATTERN": 0,
        "STRONG_TB_PATTERN": 0,
        "STAGING_PATTERN": 0
      },
      "layer3": {
        "outgoing_refs": [],
        "incoming_refs": [],
        "role_in_graph": "FS|TB|INTERMEDIATE|STAGING|UNKNOWN",
        "consolidate": false,
        "attention_boost": false,
        "aje_source_role": false,
        "path_to_tb": [],
        "path_valid": false
      },
      "layer4": {
        "passed": true,
        "blocked_by": "GATE_1|GATE_2|GATE_3|GATE_4|GATE_5|null"
      },
      "layer5_confidence": 0.0
    }
  },
  "hidden_sheets": [],
  "tb_sheets": [],
  "relationship": {
    "main_to_tb_path": [],
    "path_valid": false
  },
  "reasoning": "<one concise English sentence explaining the final decision>"
}
"""


def build_research_task_instruction(file_path: str, plan: Any = None) -> str:
    """
    ORC Research Task agent instruction.

    This agent executes the full Neural PROMAT directly against the workbook,
    including main-sheet detection, business/presentation separation,
    and Layer-7 TB/card-sheet validation.
    """
    plan_section = ""
    if plan:
        plan_section = f"""
VERIFICATION PLAN (from Plan node — follow this; NN layers override conflicts)
─────────────────────────────────────────────────────────────────────────────
{plan}
"""

    return f"""
{FULL_NN_PROMAT}

╔══════════════════════════════════════════════════════════╗
║  AGENT ROLE: Research Task Agent                        ║
╚══════════════════════════════════════════════════════════╝

You are an experienced financial accountant operating inside a LangGraph
ORC pipeline. You have direct access to Excel tools.

CORE CONSTRAINTS
────────────────
• Never invent information — every signal must be backed by a direct tool call.
• Do not trust prior agent guesses — run the NN layers yourself.
• Use tools first; draw conclusions only after observing results.
• Do not trust sheet names alone.
• Do not trust highlighted titles alone.
• Return JSON only — no markdown, no explanatory prose outside the JSON.

IMPORTANT BUSINESS RULES
────────────────────────
• The main sheet is the final financial-statement output sheet chosen by the company.
• The TB/card sheet is the structural source sheet that feeds the reporting output.
• A TB sheet is usually connected by:
    main → TB
  or:
    main → intermediate/staging → TB
• A TB sheet must be selected by structure and graph evidence, not by sheet name alone.
• Hidden sheets can NEVER be the main sheet.
• A TB/card sheet can NEVER be the final main reporting sheet.
• A staging/AJE/support sheet can NEVER be the final main reporting sheet.
• If only one strong FINAL amount column exists, treat it as FINAL even if the header is weak.
• If there is no AJE column, treat AJE = 0.
• If only FINAL exists, FINAL remains the effective amount source.

EXECUTION ORDER
───────────────
Step 1 — Open the workbook. List all sheets. Check visibility immediately.
         For hidden sheets:
           HIDDEN_SIGNAL = 1
           GATE_1 fires for main-sheet candidacy
           record them in hidden_sheets
         Still preserve the sheet name in the evidence universe if discovered.

Step 2 — Layer 0: For each visible sheet, extract workbook evidence.
         Extract at minimum:
           • sheet_name
           • headers
           • sample rows
           • formulas
           • formula references
           • flattened values
           • active-sheet status
         Also inspect likely:
           • code columns
           • description columns
           • final amount columns
           • repetition behavior
           • adjacency between code and description
           • which columns are targeted by upstream formulas

Step 3 — Layer 1: Compute the binary signals.
         Required signals:
           COA_SIGNAL
           FORMULA_SIGNAL
           CROSS_REF_SIGNAL
           REFERENCED_BY_SIGNAL
           COMPANY_COLUMN_SIGNAL
           AJE_SIGNAL
           CONSOLIDATE_SIGNAL
           HAS_CODE_COLUMN
           HAS_DESCRIPTION_COLUMN
           HAS_FINAL_COLUMN
           FINAL_REFERENCE_SIGNAL
           TB_REFERENCE_SIGNAL
           STAGING_ROLE_SIGNAL
           HIDDEN_SIGNAL

         Key interpretation rules:
           • COMPANY_COLUMN_SIGNAL is structural/business evidence, not header text only.
           • HAS_CODE_COLUMN should prefer the candidate with fewer repetitions and
             stronger adjacency to the description column.
           • HAS_DESCRIPTION_COLUMN should prefer account-description behavior,
             not generic repeated text.
           • HAS_FINAL_COLUMN may be proven by:
               - FINAL/TB-like header text
               - being the upstream formula target
               - being the dominant ending-balance numeric column
           • FINAL_REFERENCE_SIGNAL is especially important when the header is weak.
           • STAGING_ROLE_SIGNAL should turn on for AJE / adjusting / bridge /
             mapping / elimination / support / rollforward behavior.

Step 4 — Layer 2: Compute the composite patterns:
           FS_PATTERN
           PARTIAL_FS_PATTERN
           TB_PATTERN
           STRONG_TB_PATTERN
           STAGING_PATTERN

         Important:
           • TB_PATTERN must use:
               HAS_CODE_COLUMN AND HAS_DESCRIPTION_COLUMN AND HAS_FINAL_COLUMN
             and must NOT require COMPANY_COLUMN_SIGNAL = 0.
           • STRONG_TB_PATTERN is preferred for TB selection.
           • STAGING_PATTERN must be computed explicitly.

Step 5 — Layer 3: Build the cross-sheet dependency graph.
         For every sheet:
           • which sheets does it reference?
           • which sheets reference it?
         Assign:
           role_in_graph = FS | TB | INTERMEDIATE | STAGING | UNKNOWN

         Also compute:
           • consolidate
           • attention_boost
           • aje_source_role
           • path_to_tb
           • path_valid

         Important graph rules:
           • valid business direction is:
               main/presentation → intermediate/staging → TB
           • if a sheet primarily feeds AJE/adjustment columns of another sheet,
             mark aje_source_role = true
           • if a strong FS candidate has a valid path to TB, preserve it in path_to_tb

Step 6 — Layer 4: Apply the hard gates for MAIN-sheet candidacy only.
         Evaluate exactly:
           GATE_1 — hidden
           GATE_2 — no company/consolidate evidence
           GATE_3 — TB sheet
           GATE_4 — pure source
           GATE_5 — staging/AJE-support sheet

         Important:
           • GATE_2 is TECHNICAL only
           • GATE_1 / GATE_3 / GATE_4 / GATE_5 are CRITICAL
           • Do not let a blocked sheet become the technical Layer-5 winner
           • Preserve all sheet_evidence even for blocked sheets

Step 7 — Layer 5: Compute technical main-sheet confidence using TRUE SOFTMAX,
         not simple ratio normalisation.

         Use the Layer-5 logic from FULL_NN_PROMAT exactly.
         Then:
           • confidence ≥ 0.70 → main_sheet_confirmed = true
           • confidence ≥ 0.40 → main_sheet_confirmed = false but main_sheet_exists = true
           • confidence < 0.40 → main_sheet_exists = false

         technical_main_sheet = the highest-confidence valid technical winner

Step 8 — Layer 6: Perform business arbitration for the main sheet.
         Determine:
           • technical_main_sheet
           • presentation_main_sheet
           • final main_sheet_name
           • decision_mode
         Never promote:
           • hidden sheets
           • TB sheets
           • staging sheets

Step 9 — Layer 7: Select the TB/card sheet.
         Determine:
           • is_card_sheet
           • technical_tb_sheet
           • relationship.main_to_tb_path
           • relationship.path_valid

         TB selection priorities:
           1. STRONG_TB_PATTERN
           2. valid path from final main sheet
           3. better code column quality
           4. better description column quality
           5. better FINAL-column quality
           6. prefer visible evidence-backed TB sheets

Step 10 — Return the final JSON.

FILE TO ANALYSE
───────────────
{file_path}

{plan_section}

{_TASK_OUTPUT_SCHEMA}
""".strip()