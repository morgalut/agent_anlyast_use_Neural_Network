from typing import Any
from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


_TASK_OUTPUT_SCHEMA = """
REQUIRED OUTPUT — Return a single JSON object, no markdown:
{
  "main_sheet_exists": true/false,
  "main_sheet_name": "<sheet name or null>",
  "main_source_sheet_name": "<TB sheet name or null>",
  "main_sheet_confirmed": true/false,
  "confidence": 0.0,
  "is_consolidate": false,
  "nn_evidence": {
    "layer1": {
      "COA_SIGNAL": 0, "FORMULA_SIGNAL": 0, "CROSS_REF_SIGNAL": 0,
      "COMPANY_COLUMN_SIGNAL": 0, "AJE_SIGNAL": 0, "CONSOLIDATE_SIGNAL": 0,
      "CODE_COLUMN_SIGNAL": 0, "FINAL_COLUMN_SIGNAL": 0, "HIDDEN_SIGNAL": 0
    },
    "layer2": {"FS_PATTERN": 0, "TB_PATTERN": 0, "PARTIAL_FS_PATTERN": 0},
    "layer3": {
      "outgoing_refs": [], "incoming_refs": [],
      "role_in_graph": "FS|TB|INTERMEDIATE|UNKNOWN",
      "consolidate": false, "attention_boost": false
    },
    "layer4": {"passed": true, "blocked_by": null},
    "layer5_confidence": 0.0
  },
  "hidden_sheets": [],
  "tb_sheets": [],
  "reasoning": "<one concise English sentence explaining the decision>"
}
"""


def build_research_task_instruction(file_path: str, plan: Any = None) -> str:
    """
    ORC Research Task agent instruction.

    All numeric scoring and threshold-based rules have been removed.
    The agent executes the 5-layer Neural PROMAT directly against the workbook.
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
ORC pipeline.  You have direct access to Excel tools.

CORE CONSTRAINTS
────────────────
• Never invent information — every signal must be backed by a direct tool call.
• Do not trust prior agent guesses — run the NN layers yourself.
• Use tools first; draw conclusions only after observing results.
• Return JSON only — no markdown, no explanatory prose outside the JSON.

EXECUTION ORDER
───────────────
Step 1 — Open the workbook.  List all sheets.  For each sheet, check
         visibility immediately.
         HIDDEN_SIGNAL = 1 → GATE_1 fires → record in hidden_sheets,
         do NOT continue analyzing that sheet.

Step 2 — Layer 0: For each visible sheet, extract F0–F7 via OCR
         (read_only, data_only=False to capture formula strings).

Step 3 — Layer 1: Compute all 9 binary signals from the extracted features.
         Key checks:
           COA_SIGNAL  — scan all_values_flat for the 7 required sections.
           CROSS_REF_SIGNAL — check formula_sheet_refs is non-empty.
           COMPANY_COLUMN_SIGNAL — check headers for NIS/$/ USD/ILS/INC/LTD.
           CODE_COLUMN_SIGNAL — check for an alphanumeric-only column.

Step 4 — Layer 2: Compute FS_PATTERN, TB_PATTERN, PARTIAL_FS_PATTERN.
         TB_PATTERN = 1 → GATE_3 fires → record in tb_sheets.

Step 5 — Layer 3: Build the cross-sheet dependency graph.
         For every sheet X: which sheets do X's formulas reference?
         For every sheet Y: which sheets reference Y?
         Assign role_in_graph to each sheet.
         Check for CONSOLIDATE (A4) and intermediate sheets (A3).

Step 6 — Layer 4: Apply GATE_1 through GATE_4 to each remaining sheet.
         GATE_2 is critical — block any sheet with no company columns
         that is not a confirmed CONSOLIDATE.

Step 7 — Layer 5: Compute signal strength S(sheet) and softmax confidence
         for all sheets that passed Layer 4.
         confidence ≥ 0.70 → main_sheet_confirmed = true
         confidence ≥ 0.40 → main_sheet_confirmed = false (report uncertainty)
         confidence < 0.40 → main_sheet_exists = false

Step 8 — Identify main_source_sheet_name from Layer-3 A2
         (the sheet most frequently referenced by the FS candidate).

Step 9 — Return the output JSON.

FILE TO ANALYSE
───────────────
{file_path}

{plan_section}

{_TASK_OUTPUT_SCHEMA}
""".strip()