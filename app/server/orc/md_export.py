from __future__ import annotations

import json
import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def safe_preview(value: Any, limit: int = 3000) -> Any:
    try:
        text = (
            json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, (dict, list))
            else str(value)
        )
        return (
            (text[:limit] + "...<truncated>")
            if len(text) > limit
            else (value if isinstance(value, (dict, list)) else text)
        )
    except Exception:
        return str(value)


def ensure_debug_trace(state) -> Dict[str, Any]:
    trace = state.get("debug_trace")
    if trace:
        return trace
    trace = {
        "run_id":     str(uuid.uuid4()),
        "started_at": utc_now(),
        "status":     "running",
        "input":      {"user_input": state.get("user_input")},
        "steps": [], "agents": [], "court_sessions": [], "errors": [],
    }
    state["debug_trace"] = trace
    return trace


def _parse_json_from_text(text: str) -> dict | None:
    """Extract the first JSON object from arbitrary LLM text output."""
    if not text:
        return None
    clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def _extract_nn_evidence(state, sheet_name: str | None) -> dict:
    """
    Pull NN layer evidence for a specific sheet_name from task_results.

    Reads the new NN PROMAT output schema:
      nn_evidence.layer1.COMPANY_COLUMN_SIGNAL  (0/1)
      nn_evidence.layer1.HIDDEN_SIGNAL          (0/1)
      nn_evidence.layer1.COA_SIGNAL             (0/1)
      nn_evidence.layer2.FS_PATTERN             (0/1)
      nn_evidence.layer2.TB_PATTERN             (0/1)
      nn_evidence.layer3.role_in_graph          (str)
      nn_evidence.layer4.passed                 (bool)
      nn_evidence.layer4.blocked_by             (str|null)
      nn_evidence.layer5_confidence             (float 0-1)
      confidence                                (float 0-1)

    Returns an empty dict if no matching task result is found.
    """
    if not sheet_name:
        return {}

    for task in state.get("task_results", []):
        parsed = _parse_json_from_text(task.get("result", ""))
        if not parsed or parsed.get("main_sheet_name") != sheet_name:
            continue

        nn = parsed.get("nn_evidence", {})
        l1 = nn.get("layer1", {})
        l2 = nn.get("layer2", {})
        l3 = nn.get("layer3", {})
        l4 = nn.get("layer4", {})

        return {
            # Layer 1 signals
            "COMPANY_COLUMN_SIGNAL": int(l1.get("COMPANY_COLUMN_SIGNAL", 0)),
            "HIDDEN_SIGNAL":         int(l1.get("HIDDEN_SIGNAL", 0)),
            "COA_SIGNAL":            int(l1.get("COA_SIGNAL", 0)),
            "CONSOLIDATE_SIGNAL":    int(l1.get("CONSOLIDATE_SIGNAL", 0)),
            "CROSS_REF_SIGNAL":      int(l1.get("CROSS_REF_SIGNAL", 0)),
            "AJE_SIGNAL":            int(l1.get("AJE_SIGNAL", 0)),
            "FORMULA_SIGNAL":        int(l1.get("FORMULA_SIGNAL", 0)),
            "CODE_COLUMN_SIGNAL":    int(l1.get("CODE_COLUMN_SIGNAL", 0)),
            "FINAL_COLUMN_SIGNAL":   int(l1.get("FINAL_COLUMN_SIGNAL", 0)),
            # Layer 2 patterns
            "FS_PATTERN":            int(l2.get("FS_PATTERN", 0)),
            "TB_PATTERN":            int(l2.get("TB_PATTERN", 0)),
            "PARTIAL_FS_PATTERN":    int(l2.get("PARTIAL_FS_PATTERN", 0)),
            # Layer 3 graph
            "role_in_graph":         l3.get("role_in_graph", "UNKNOWN"),
            "consolidate":           bool(l3.get("consolidate", False)),
            # Layer 4 gates
            "gate_passed":           bool(l4.get("passed", True)),
            "blocked_by":            l4.get("blocked_by"),
            # Layer 5 confidence — already float 0-1 in new schema
            "confidence":            float(nn.get("layer5_confidence", 0.0)
                                           or parsed.get("confidence", 0.0)),
        }

    return {}


def _format_candidate_block(candidates: list, title: str) -> str:
    lines = [f"## {title}", ""]
    if not candidates:
        return "\n".join(lines + ["No candidates available.", ""])
    for i, item in enumerate(candidates, 1):
        lines += [
            f"{i}. **{item.get('sheet')}**",
            f"   - Heuristic score: `{item.get('score')}`",
        ]
        if item.get("title"):
            lines.append(f"   - Detected title: `{item['title']}`")
    lines.append("")
    return "\n".join(lines)


def _format_nn_evidence(ev: dict) -> list[str]:
    """Render NN layer signals for the markdown report."""
    if not ev:
        return ["  *No NN evidence available for this sheet.*"]
    lines = []
    # Layer 1
    l1_signals = {
        "COA_SIGNAL":            ev.get("COA_SIGNAL"),
        "COMPANY_COLUMN_SIGNAL": ev.get("COMPANY_COLUMN_SIGNAL"),
        "CROSS_REF_SIGNAL":      ev.get("CROSS_REF_SIGNAL"),
        "FORMULA_SIGNAL":        ev.get("FORMULA_SIGNAL"),
        "HIDDEN_SIGNAL":         ev.get("HIDDEN_SIGNAL"),
        "AJE_SIGNAL":            ev.get("AJE_SIGNAL"),
        "CONSOLIDATE_SIGNAL":    ev.get("CONSOLIDATE_SIGNAL"),
        "CODE_COLUMN_SIGNAL":    ev.get("CODE_COLUMN_SIGNAL"),
        "FINAL_COLUMN_SIGNAL":   ev.get("FINAL_COLUMN_SIGNAL"),
    }
    fired  = [k for k, v in l1_signals.items() if v == 1]
    silent = [k for k, v in l1_signals.items() if v == 0]
    if fired:
        lines.append(f"  - **L1 active signals**: `{'` `'.join(fired)}`")
    if silent:
        lines.append(f"  - L1 silent signals: `{'` `'.join(silent)}`")
    # Layer 2
    if ev.get("FS_PATTERN") is not None:
        pat = []
        if ev.get("FS_PATTERN"):        pat.append("FS_PATTERN ✓")
        if ev.get("TB_PATTERN"):        pat.append("TB_PATTERN ✓")
        if ev.get("PARTIAL_FS_PATTERN"): pat.append("PARTIAL_FS_PATTERN ✓")
        lines.append(f"  - L2 patterns: {', '.join(pat) or 'none fired'}")
    # Layer 3
    if ev.get("role_in_graph"):
        lines.append(f"  - L3 graph role: `{ev['role_in_graph']}`")
    # Layer 4
    gate_status = "✓ passed" if ev.get("gate_passed") else f"✗ blocked ({ev.get('blocked_by')})"
    lines.append(f"  - L4 gate: {gate_status}")
    # Layer 5
    if ev.get("confidence") is not None:
        lines.append(f"  - L5 confidence: `{ev['confidence']:.3f}`")
    return lines


def _format_business_arbitration_block(ba: dict) -> list[str]:
    """Render the Layer-6 business arbitration summary for the markdown report."""
    if not ba:
        return ["  *No business arbitration data available.*"]
    lines = []
    lines.append(f"  - Technical winner sheet type: `{ba.get('technical_winner_sheet_type', 'N/A')}`")
    bc = ba.get("business_candidate")
    if bc:
        lines.append(f"  - Business candidate: **`{bc}`**")
        lines.append(f"    - Type: `{ba.get('business_candidate_sheet_type', 'N/A')}`")
        lines.append(f"    - Blocked by: `{ba.get('business_candidate_blocked_by') or 'none'}`")
        lines.append(f"    - Disqualification class: `{ba.get('business_candidate_disqualification_class', 'N/A')}`")
    else:
        lines.append("  - No REPORTING_FS business candidate found")
    override = ba.get("override_applied", False)
    lines.append(f"  - Override applied: **{'✅ YES' if override else '❌ NO'}**")
    return lines


def _build_decision_markdown(state) -> str:
    result         = state.get("main_sheet_result", {})
    chosen         = state.get("main_sheet_name")
    has_main       = state.get("has_main_sheet", False)
    detector_cand  = state.get("detector_candidate")
    active_sheet   = result.get("active_sheet")

    # Parse final answer for L6 fields
    final_answer_raw = state.get("final_answer", "")
    final_parsed     = _parse_json_from_text(final_answer_raw) or {}
    technical_main   = final_parsed.get("technical_main_sheet")
    business_main    = final_parsed.get("business_main_sheet")
    decision_mode    = final_parsed.get("decision_mode", "unknown")
    ba_block         = final_parsed.get("business_arbitration", {})
    main_source_nm   = final_parsed.get("main_source_sheet") if isinstance(final_parsed, dict) else None

    out_cands  = result.get("output_candidates", [])
    src_cands  = result.get("source_candidates", [])
    agent_result = (state.get("task_results") or [{}])[0].get("result", "")

    # Pull NN evidence for the chosen sheet
    chosen_ev = _extract_nn_evidence(state, chosen)

    lines = [
        "# Main Sheet Decision Report", "",
        f"- Run time: `{utc_now()}`",
        f"- Workbook: `{state.get('user_input')}`",
        f"- **Main sheet found: `{has_main}`**",
        f"- **Final main sheet: `{chosen}`**  ← synthesis + L6 authoritative",
        f"- **Technical main sheet: `{technical_main}`**  ← L5 softmax winner",
        f"- **Business main sheet: `{business_main}`**  ← L6 business candidate",
        f"- **Decision mode: `{decision_mode}`**",
        f"- Detector candidate (hint only): `{detector_cand}`",
        f"- Strongest source sheet: `{main_source_nm}`",
        f"- Workbook active sheet: `{active_sheet}`",
        "",
    ]

    # Decision mode banner
    if decision_mode == "business_override":
        lines += [
            "🔄 **Business Override Active**: Layer-6 arbitration detected a mismatch "
            "between the technical winner (computational hub) and the business-correct "
            "final output sheet. The REPORTING_FS sheet was promoted.",
            "",
        ]
    elif decision_mode == "technical_default":
        lines += [
            "✅ **Technical Default**: Layer-6 confirmed that the technical winner "
            "is also the business-correct final sheet. No override needed.",
            "",
        ]

    # Detector override notice
    if detector_cand and detector_cand != chosen:
        lines += [
            "⚠️ **Detector override**: the heuristic detector suggested "
            f"`{detector_cand}` but the NN PROMAT + synthesis chose `{chosen}`.",
            "This is correct — the detector is a hint generator, not a decision maker.",
            "",
        ]

    # Court sessions
    sessions = ensure_debug_trace(state).get("court_sessions", [])
    if sessions:
        lines += ["## Court Sessions", ""]
        for cs in sessions:
            lines.append(
                f"- `{cs['agent_under_review']}` | "
                f"Attempt {cs['attempt']} | **{cs['verdict']}**"
            )
        lines.append("")

    # Why this sheet — NN evidence block
    lines += ["## Why this sheet was chosen", ""]
    if chosen:
        lines.append(
            f"Synthesis + Layer-6 selected **`{chosen}`** based on NN PROMAT layer evidence:"
        )
        lines += _format_nn_evidence(chosen_ev)
    else:
        lines.append(
            "No main sheet identified — all candidates were blocked by NN gates "
            "or Python guardrails."
        )
        if detector_cand:
            det_ev = _extract_nn_evidence(state, detector_cand)
            lines.append(
                f"The detector suggested `{detector_cand}` but it failed "
                f"gate checks: {det_ev.get('blocked_by', 'unknown')}."
            )
    lines.append("")

    # Layer 6 business arbitration block
    lines += ["## Layer-6 Business Arbitration", ""]
    lines += _format_business_arbitration_block(ba_block)
    lines.append("")

    # Authority chain
    lines += [
        "## Decision authority chain", "",
        "```",
        "L6 Business Arbitration (semantic bridge)",
        "      ↑",
        "L5 Softmax (technical winner)",
        "      ↑",
        "L4 Hard gates (GATE_1–GATE_4)",
        "      ↑",
        "L3 Context graph",
        "      ↑",
        "L2 Pattern logic",
        "      ↑",
        "L1 Binary signals",
        "      ↑",
        "L0 Raw extraction",
        "      ↑",
        "Detector (heuristic hint — never decides alone)",
        "```",
        "",
    ]

    # Research agent NN output
    if agent_result:
        lines += [
            "## Research agent NN output", "",
            "```json", str(agent_result), "```", "",
        ]

    # Candidate rankings (heuristic scores kept for reference)
    lines += ["## Candidate rankings (heuristic — for reference only)", ""]
    lines.append(_format_candidate_block(out_cands, "Top output candidates"))
    lines.append(_format_candidate_block(src_cands, "Top source candidates"))

    # Final synthesis JSON
    if state.get("final_answer"):
        lines += ["## Final synthesis JSON", "", "```json", str(state["final_answer"]), "```", ""]

    return "\n".join(lines).strip() + "\n"




def export_artifacts(state) -> tuple[str, str]:
    trace = ensure_debug_trace(state)
    trace["finished_at"] = utc_now()
    trace["status"]      = "success"

    export_dir = Path("debug_traces")
    export_dir.mkdir(parents=True, exist_ok=True)
    run_id = trace["run_id"]

    payload = {
        "run_id":             run_id,
        "started_at":         trace["started_at"],
        "finished_at":        trace["finished_at"],
        "status":             trace["status"],
        "input":              {"file_path": state.get("user_input")},
        "has_main_sheet":     state.get("has_main_sheet", False),
        "main_sheet_name":    state.get("main_sheet_name"),
        "detector_candidate": state.get("detector_candidate"),
        "excel_summary":      state.get("excel_summary"),
        "main_sheet_result":  state.get("main_sheet_result"),
        "analysis":           state.get("analysis"),
        "plan":               state.get("plan"),
        "task_results":       state.get("task_results"),
        "court_sessions":     trace.get("court_sessions", []),
        "final_answer":       state.get("final_answer"),
        "steps":              trace.get("steps", []),
        "agents":             trace.get("agents", []),
        "errors":             trace.get("errors", []),
    }

    json_file = export_dir / f"{run_id}.json"
    md_file   = export_dir / f"{run_id}.md"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(_build_decision_markdown(state))

    state["export_file"] = state["debug_trace_file"] = str(json_file)
    state["md_export_file"] = str(md_file)
    return str(json_file), str(md_file)
