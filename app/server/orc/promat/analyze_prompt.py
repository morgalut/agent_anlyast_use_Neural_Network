from typing import Any
from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


def build_analyze_prompt(excel_summary: Any, main_sheet_result: Any) -> str:
    """
    ORC Analyze-node entry point.

    The old additive scoring system has been replaced by the 5-layer
    Neural PROMAT.  This agent runs all five layers and returns structured
    evidence, not a numeric score.
    """
    return f"""
{FULL_NN_PROMAT}

╔══════════════════════════════════════════════════════════╗
║  AGENT ROLE: ORC Analyze Node                           ║
╚══════════════════════════════════════════════════════════╝

You are the Analyze node in a LangGraph ORC pipeline.
The input is an Excel workbook path — not a free-text query.

YOUR TASK
─────────
For every sheet listed in the workbook summary below:

  1. Run Layer 0  — extract F0–F7 from the summary data.
  2. Run Layer 1  — compute all 9 binary signals.
  3. Run Layer 2  — compute FS_PATTERN, TB_PATTERN, PARTIAL_FS_PATTERN.
  4. Run Layer 3  — build the cross-sheet dependency graph;
                    assign role_in_graph; detect CONSOLIDATE sheets.
  5. Run Layer 4  — apply GATE_1 through GATE_4; block disqualified sheets.
  6. Run Layer 5  — compute softmax confidence over passing sheets.
  7. Select the strongest candidate.

ADDITIONAL RULES
────────────────
• The detector result (main_sheet_result) is a HINT only.
  Do not accept it without passing it through all 5 NN layers.
• A hidden sheet always triggers GATE_1 — disqualified immediately.
• A sheet with TB_PATTERN = 1 always triggers GATE_3 — disqualified.
• A sheet with no company columns and no CONSOLIDATE signal triggers GATE_2.
  This is the most common source of wrong answers (CF-type errors).

LIVE INPUT
──────────
Workbook summary:
{excel_summary}

Detector hint (Layer 0 starting candidate — verify through NN):
{main_sheet_result}

REQUIRED OUTPUT — Return a single JSON object, no markdown, no explanation:
{{
  "main_sheet_exists": true/false,
  "strongest_candidate": "<sheet name or null>",
  "confidence": 0.0,
  "reasoning": "<one concise English sentence>",
  "nn_layers": {{
    "layer1_signals": {{
      "<sheet>": {{
        "COA_SIGNAL": 0, "FORMULA_SIGNAL": 0, "CROSS_REF_SIGNAL": 0,
        "COMPANY_COLUMN_SIGNAL": 0, "AJE_SIGNAL": 0,
        "CONSOLIDATE_SIGNAL": 0, "CODE_COLUMN_SIGNAL": 0,
        "FINAL_COLUMN_SIGNAL": 0, "HIDDEN_SIGNAL": 0
      }}
    }},
    "layer2_patterns": {{
      "<sheet>": {{"FS_PATTERN": 0, "TB_PATTERN": 0, "PARTIAL_FS_PATTERN": 0}}
    }},
    "layer3_graph": {{
      "<sheet>": {{
        "outgoing_refs": [], "incoming_refs": [],
        "role_in_graph": "FS|TB|INTERMEDIATE|UNKNOWN",
        "consolidate": false, "attention_boost": false
      }}
    }},
    "layer4_gates": {{
      "<sheet>": {{"passed": true, "blocked_by": null}}
    }},
    "layer5_confidence": {{
      "<sheet>": 0.0
    }}
  }},
  "blocked_sheets": {{
    "hidden": [], "tb": [], "no_company": [], "incoming_only": []
  }},
  "next_research_steps": ["<what the research agent should verify>"]
}}
""".strip()