from app.server.orc.promat.nn_promat_core import FULL_NN_PROMAT


# ──────────────────────────────────────────────────────────────────────────────
#  Ready-to-use Python helpers implementing all 5 NN layers.
#  The Coder Agent generates code using these patterns.
# ──────────────────────────────────────────────────────────────────────────────

_NN_PYTHON_HELPERS = '''
# ── Layer 0+1: signal extraction ─────────────────────────────────────────────
import re
from openpyxl import load_workbook

REQUIRED_SECTIONS = frozenset([
    "assets", "current assets", "long-term assets",
    "liabilities and equity", "current liabilities",
    "long-term liabilities", "equity",
])
COMPANY_KW   = frozenset(["nis", "dollar", "$", "usd", "ils", "inc", "ltd"])
AJE_KW       = frozenset(["aje", "adjusting"])
CONSOL_KW    = frozenset(["consolidate", "consol", "total"])
FINAL_KW     = frozenset(["final", "trial balance", " tb "])
CODE_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
SHEET_REF    = re.compile(r"\'([^\']+)\'!")


def extract_signals(file_path: str, sheet_name: str, wb_full) -> dict:
    """Open sheet with formulas, extract Layer-0 features, compute Layer-1 signals."""
    hidden = wb_full.get_sheet_visibility(sheet_name) in ("hidden", "veryHidden")

    wb_f = load_workbook(file_path, read_only=True, data_only=False)
    ws   = wb_f[sheet_name]

    headers, formulas, refs, flat = [], [], [], []

    for r_idx, row in enumerate(ws.iter_rows(values_only=False), 1):
        for cell in row:
            v = str(cell.value or "").strip()
            if not v:
                continue
            flat.append(v.lower())
            if v.startswith("="):
                formulas.append(v)
                refs.extend(SHEET_REF.findall(v))
        if r_idx == 1:
            headers = [str(c.value or "").strip() for c in row if c.value]
        if len(flat) >= 300:
            break

    wb_f.close()

    flat_set    = set(flat)
    hdr_lower   = [h.lower() for h in headers]
    coa_hits    = sum(1 for s in REQUIRED_SECTIONS if s in flat_set)
    code_vals   = [v for v in flat if CODE_PATTERN.match(v) and len(v) <= 12]

    return {
        # Layer 1 signals — all binary
        "HIDDEN_SIGNAL":          int(hidden),
        "COA_SIGNAL":             int(coa_hits >= 3),
        "FORMULA_SIGNAL":         int(bool(formulas)),
        "CROSS_REF_SIGNAL":       int(bool(refs)),
        "COMPANY_COLUMN_SIGNAL":  int(any(kw in h for kw in COMPANY_KW for h in hdr_lower)),
        "AJE_SIGNAL":             int(any(kw in h for kw in AJE_KW    for h in hdr_lower)),
        "CONSOLIDATE_SIGNAL":     int(any(kw in h for kw in CONSOL_KW for h in hdr_lower)),
        "CODE_COLUMN_SIGNAL":     int(len(code_vals) >= 5 and coa_hits == 0),
        "FINAL_COLUMN_SIGNAL":    int(any(kw in h for kw in FINAL_KW  for h in hdr_lower)),
        # metadata for Layer 3
        "_outgoing_refs": list(set(refs)),
        "_coa_hits":      coa_hits,
        "_formulas":      formulas[:20],
        "_headers":       headers,
    }


# ── Layer 2: pattern computation ─────────────────────────────────────────────
def compute_patterns(sig: dict) -> dict:
    fs = (
        sig["COA_SIGNAL"] and sig["FORMULA_SIGNAL"] and sig["CROSS_REF_SIGNAL"]
        and sig["COMPANY_COLUMN_SIGNAL"]
        and not sig["HIDDEN_SIGNAL"] and not sig["CODE_COLUMN_SIGNAL"]
    )
    tb = (
        sig["CODE_COLUMN_SIGNAL"] and sig["FINAL_COLUMN_SIGNAL"]
        and not sig["COA_SIGNAL"] and not sig["COMPANY_COLUMN_SIGNAL"]
    )
    partial = (
        sig["COA_SIGNAL"]
        and (sig["FORMULA_SIGNAL"] or sig["CROSS_REF_SIGNAL"])
        and not sig["HIDDEN_SIGNAL"]
        and not tb
    )
    return {
        "FS_PATTERN":      int(bool(fs)),
        "TB_PATTERN":      int(bool(tb)),
        "PARTIAL_FS_PATTERN": int(bool(partial)),
    }


# ── Layer 3: dependency graph ─────────────────────────────────────────────────
def build_graph(all_signals: dict) -> dict:
    """
    all_signals: { sheet_name: signals_dict }
    Returns: { sheet_name: { outgoing_refs, incoming_refs, role_in_graph, ... } }
    """
    graph = {sn: {"outgoing_refs": sig["_outgoing_refs"], "incoming_refs": [],
                  "role_in_graph": "UNKNOWN", "consolidate": False,
                  "attention_boost": False}
             for sn, sig in all_signals.items()}

    # Populate incoming_refs
    for sn, data in graph.items():
        for ref in data["outgoing_refs"]:
            if ref in graph:
                graph[ref]["incoming_refs"].append(sn)

    # Assign roles
    for sn, data in graph.items():
        has_out = bool(data["outgoing_refs"])
        has_in  = bool(data["incoming_refs"])
        pat     = compute_patterns(all_signals[sn])
        if pat["TB_PATTERN"]:
            data["role_in_graph"] = "TB"
        elif pat["FS_PATTERN"] or (has_out and not has_in):
            data["role_in_graph"] = "FS"
        elif has_in and not has_out:
            data["role_in_graph"] = "TB"
        elif has_out and has_in:
            data["role_in_graph"] = "INTERMEDIATE"

    return graph


# ── Layer 4: hard gates ───────────────────────────────────────────────────────
def apply_gates(sig: dict, pat: dict, graph_node: dict) -> dict:
    if sig["HIDDEN_SIGNAL"]:
        return {"passed": False, "blocked_by": "GATE_1"}
    consolidate_exempt = graph_node.get("consolidate", False)
    if not sig["COMPANY_COLUMN_SIGNAL"] and not sig["CONSOLIDATE_SIGNAL"] and not consolidate_exempt:
        return {"passed": False, "blocked_by": "GATE_2"}
    if pat["TB_PATTERN"]:
        return {"passed": False, "blocked_by": "GATE_3"}
    if (graph_node["role_in_graph"] == "TB"
            and not pat["FS_PATTERN"] and not pat["PARTIAL_FS_PATTERN"]):
        return {"passed": False, "blocked_by": "GATE_4"}
    return {"passed": True, "blocked_by": None}


# ── Layer 5: softmax confidence ───────────────────────────────────────────────
def signal_strength(sig: dict, pat: dict, attention_boost: bool = False) -> float:
    return (
        pat["FS_PATTERN"]          * 1.00
        + pat["PARTIAL_FS_PATTERN"]  * 0.50
        + sig["AJE_SIGNAL"]          * 0.20
        + sig["CONSOLIDATE_SIGNAL"]  * 0.15
        + (0.15 if attention_boost else 0)
        + sig["CROSS_REF_SIGNAL"]    * 0.10
    )


def softmax_confidence(strengths: dict) -> dict:
    total = sum(strengths.values()) or 1e-9
    return {s: round(v / total, 4) for s, v in strengths.items()}
'''


_CODER_OUTPUT_SCHEMA = """
REQUIRED OUTPUT — Return a single JSON object, no markdown:
{
  "main_sheet_exists": true/false,
  "main_sheet_name": "<sheet name or null>",
  "main_source_sheet_name": "<TB sheet name or null>",
  "main_sheet_confirmed": true/false,
  "confidence": 0.0,
  "is_consolidate": false,
  "nn_evidence": {
    "layer1": { ... all 9 signals ... },
    "layer2": { "FS_PATTERN": 0, "TB_PATTERN": 0, "PARTIAL_FS_PATTERN": 0 },
    "layer3": { "outgoing_refs": [], "incoming_refs": [],
                "role_in_graph": "FS", "consolidate": false },
    "layer4": { "passed": true, "blocked_by": null },
    "layer5_confidence": 0.0
  },
  "hidden_sheets": [],
  "tb_sheets": [],
  "reasoning": "<one concise English sentence>"
}
"""


def build_coder_system_prompt() -> str:
    """
    ORC Coder Agent system prompt.

    ReAct behavior + 5-layer Neural PROMAT.
    All scoring dicts and numeric penalties have been removed.
    Ready-to-run Python helpers are provided for each NN layer.
    """
    return f"""
{FULL_NN_PROMAT}

╔══════════════════════════════════════════════════════════╗
║  AGENT ROLE: Coder Agent (ReAct + Neural PROMAT)        ║
╚══════════════════════════════════════════════════════════╝

You are the Coder Agent in a LangGraph ORC pipeline.
Use ReAct behavior: Reason → generate code → execute → observe → repeat.

CORE CONSTRAINTS
────────────────
• Generate code only when it adds new information not already observed.
• Every code block must implement the 5-layer NN and return valid JSON.
• No numeric scoring, no additive weights, no penalty constants.
• All decisions flow through binary signals → pattern logic → gates → softmax.

PYTHON HELPERS (use these directly in generated code)
──────────────────────────────────────────────────────
{_NN_PYTHON_HELPERS}

EXECUTION ORDER FOR GENERATED CODE
────────────────────────────────────
1. Load workbook (read_only=True).  Enumerate sheets.
2. For each sheet: check visibility → HIDDEN_SIGNAL.
   If HIDDEN_SIGNAL = 1 → GATE_1 → skip to next sheet.
3. Call extract_signals(file_path, sheet_name, wb_full) → Layer 1.
4. Call compute_patterns(sig) → Layer 2.
   If TB_PATTERN = 1 → GATE_3 → add to tb_sheets, skip.
5. After all sheets processed: call build_graph(all_signals) → Layer 3.
6. For each remaining sheet: call apply_gates(sig, pat, graph_node) → Layer 4.
7. For passing sheets: call signal_strength → softmax_confidence → Layer 5.
8. Select winner by highest confidence.
   confidence ≥ 0.70 → main_sheet_confirmed = true.
   confidence ≥ 0.40 → main_sheet_confirmed = false.
   confidence < 0.40 → main_sheet_exists = false.
9. Return single JSON object.

{_CODER_OUTPUT_SCHEMA}
""".strip()