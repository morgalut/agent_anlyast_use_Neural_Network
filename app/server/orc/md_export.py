from __future__ import annotations

import json
import re
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
        "run_id": str(uuid.uuid4()),
        "started_at": utc_now(),
        "status": "running",
        "input": {"user_input": state.get("user_input")},
        "steps": [],
        "agents": [],
        "court_sessions": [],
        "errors": [],
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

    Supports the updated 7-layer PROMAT schema and legacy-compatible fallbacks.
    Returns an empty dict if no matching sheet evidence is found.
    """
    if not sheet_name:
        return {}

    for task in state.get("task_results", []):
        parsed = _parse_json_from_text(task.get("result", ""))
        if not parsed:
            continue

        # Preferred: per-sheet evidence map
        sheet_evidence = parsed.get("sheet_evidence", {}) or {}
        payload = sheet_evidence.get(sheet_name)

        # Legacy fallback: nn_evidence only for chosen main sheet
        if not payload and parsed.get("main_sheet_name") == sheet_name:
            payload = parsed.get("nn_evidence", {})

        if not payload:
            continue

        l1 = payload.get("layer1", {}) or {}
        l2 = payload.get("layer2", {}) or {}
        l3 = payload.get("layer3", {}) or {}
        l4 = payload.get("layer4", {}) or {}

        return {
            # Layer 1
            "COA_SIGNAL": int(l1.get("COA_SIGNAL", 0)),
            "FORMULA_SIGNAL": int(l1.get("FORMULA_SIGNAL", 0)),
            "CROSS_REF_SIGNAL": int(l1.get("CROSS_REF_SIGNAL", 0)),
            "REFERENCED_BY_SIGNAL": int(l1.get("REFERENCED_BY_SIGNAL", 0)),
            "COMPANY_COLUMN_SIGNAL": int(l1.get("COMPANY_COLUMN_SIGNAL", 0)),
            "AJE_SIGNAL": int(l1.get("AJE_SIGNAL", 0)),
            "CONSOLIDATE_SIGNAL": int(l1.get("CONSOLIDATE_SIGNAL", 0)),
            "HAS_CODE_COLUMN": int(l1.get("HAS_CODE_COLUMN", l1.get("CODE_COLUMN_SIGNAL", 0))),
            "HAS_DESCRIPTION_COLUMN": int(l1.get("HAS_DESCRIPTION_COLUMN", 0)),
            "HAS_FINAL_COLUMN": int(l1.get("HAS_FINAL_COLUMN", l1.get("FINAL_COLUMN_SIGNAL", 0))),
            "FINAL_REFERENCE_SIGNAL": int(l1.get("FINAL_REFERENCE_SIGNAL", 0)),
            "TB_REFERENCE_SIGNAL": int(l1.get("TB_REFERENCE_SIGNAL", 0)),
            "STAGING_ROLE_SIGNAL": int(l1.get("STAGING_ROLE_SIGNAL", 0)),
            "HIDDEN_SIGNAL": int(l1.get("HIDDEN_SIGNAL", 0)),

            # Backward-compatible aliases for display convenience
            "CODE_COLUMN_SIGNAL": int(l1.get("CODE_COLUMN_SIGNAL", l1.get("HAS_CODE_COLUMN", 0))),
            "FINAL_COLUMN_SIGNAL": int(l1.get("FINAL_COLUMN_SIGNAL", l1.get("HAS_FINAL_COLUMN", 0))),

            # Layer 2
            "FS_PATTERN": int(l2.get("FS_PATTERN", 0)),
            "TB_PATTERN": int(l2.get("TB_PATTERN", 0)),
            "PARTIAL_FS_PATTERN": int(l2.get("PARTIAL_FS_PATTERN", 0)),
            "STRONG_TB_PATTERN": int(l2.get("STRONG_TB_PATTERN", 0)),
            "STAGING_PATTERN": int(l2.get("STAGING_PATTERN", 0)),

            # Layer 3
            "role_in_graph": l3.get("role_in_graph", "UNKNOWN"),
            "consolidate": bool(l3.get("consolidate", False)),
            "attention_boost": bool(l3.get("attention_boost", False)),
            "aje_source_role": bool(l3.get("aje_source_role", False)),
            "outgoing_refs": list(l3.get("outgoing_refs", []) or []),
            "incoming_refs": list(l3.get("incoming_refs", []) or []),
            "path_to_tb": list(l3.get("path_to_tb", []) or []),
            "path_valid": bool(l3.get("path_valid", False)),

            # Layer 4
            "gate_passed": bool(l4.get("passed", True)),
            "blocked_by": l4.get("blocked_by"),

            # Layer 5
            "confidence": float(
                payload.get("layer5_confidence", 0.0) or parsed.get("confidence", 0.0)
            ),
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

    l1_signals = {
        "COA_SIGNAL": ev.get("COA_SIGNAL"),
        "FORMULA_SIGNAL": ev.get("FORMULA_SIGNAL"),
        "CROSS_REF_SIGNAL": ev.get("CROSS_REF_SIGNAL"),
        "REFERENCED_BY_SIGNAL": ev.get("REFERENCED_BY_SIGNAL"),
        "COMPANY_COLUMN_SIGNAL": ev.get("COMPANY_COLUMN_SIGNAL"),
        "AJE_SIGNAL": ev.get("AJE_SIGNAL"),
        "CONSOLIDATE_SIGNAL": ev.get("CONSOLIDATE_SIGNAL"),
        "HAS_CODE_COLUMN": ev.get("HAS_CODE_COLUMN"),
        "HAS_DESCRIPTION_COLUMN": ev.get("HAS_DESCRIPTION_COLUMN"),
        "HAS_FINAL_COLUMN": ev.get("HAS_FINAL_COLUMN"),
        "FINAL_REFERENCE_SIGNAL": ev.get("FINAL_REFERENCE_SIGNAL"),
        "TB_REFERENCE_SIGNAL": ev.get("TB_REFERENCE_SIGNAL"),
        "STAGING_ROLE_SIGNAL": ev.get("STAGING_ROLE_SIGNAL"),
        "HIDDEN_SIGNAL": ev.get("HIDDEN_SIGNAL"),
    }

    fired = [k for k, v in l1_signals.items() if v == 1]
    silent = [k for k, v in l1_signals.items() if v == 0]

    if fired:
        lines.append(f"  - **L1 active signals**: `{'` `'.join(fired)}`")
    if silent:
        lines.append(f"  - L1 silent signals: `{'` `'.join(silent)}`")

    pat = []
    if ev.get("FS_PATTERN"):
        pat.append("FS_PATTERN ✓")
    if ev.get("TB_PATTERN"):
        pat.append("TB_PATTERN ✓")
    if ev.get("PARTIAL_FS_PATTERN"):
        pat.append("PARTIAL_FS_PATTERN ✓")
    if ev.get("STRONG_TB_PATTERN"):
        pat.append("STRONG_TB_PATTERN ✓")
    if ev.get("STAGING_PATTERN"):
        pat.append("STAGING_PATTERN ✓")
    lines.append(f"  - L2 patterns: {', '.join(pat) or 'none fired'}")

    lines.append(f"  - L3 graph role: `{ev.get('role_in_graph', 'UNKNOWN')}`")
    if ev.get("outgoing_refs"):
        lines.append(f"  - L3 outgoing refs: `{', '.join(ev['outgoing_refs'])}`")
    if ev.get("incoming_refs"):
        lines.append(f"  - L3 incoming refs: `{', '.join(ev['incoming_refs'])}`")
    if ev.get("path_to_tb"):
        lines.append(f"  - L3 path to TB: `{ ' → '.join(ev['path_to_tb']) }`")
    lines.append(f"  - L3 path valid: `{ev.get('path_valid', False)}`")
    if ev.get("consolidate", False):
        lines.append("  - L3 consolidate flag: `true`")
    if ev.get("attention_boost", False):
        lines.append("  - L3 attention boost: `true`")
    if ev.get("aje_source_role", False):
        lines.append("  - L3 AJE source role: `true`")

    gate_status = "✓ passed" if ev.get("gate_passed") else f"✗ blocked ({ev.get('blocked_by')})"
    lines.append(f"  - L4 gate: {gate_status}")

    if ev.get("confidence") is not None:
        lines.append(f"  - L5 confidence: `{ev['confidence']:.3f}`")

    return lines


def _format_business_arbitration_block(ba: dict) -> list[str]:
    """Render the Layer-6 business arbitration summary for the markdown report."""
    if not ba:
        return ["  *No business arbitration data available.*"]

    lines = []
    lines.append(
        f"  - Technical winner sheet type: `{ba.get('technical_winner_sheet_type', 'N/A')}`"
    )

    pc = ba.get("presentation_candidate") or ba.get("business_candidate")
    if pc:
        lines.append(f"  - Presentation candidate: **`{pc}`**")
        lines.append(
            f"    - Type: `{ba.get('presentation_candidate_sheet_type', ba.get('business_candidate_sheet_type', 'N/A'))}`"
        )
        lines.append(
            f"    - Blocked by: `{ba.get('presentation_candidate_blocked_by', ba.get('business_candidate_blocked_by')) or 'none'}`"
        )
        lines.append(
            f"    - Disqualification class: `{ba.get('presentation_candidate_disqualification_class', ba.get('business_candidate_disqualification_class', 'N/A'))}`"
        )
    else:
        lines.append("  - No REPORTING_FS presentation candidate found")

    override = ba.get("override_applied", False)
    lines.append(f"  - Override applied: **{'✅ YES' if override else '❌ NO'}**")
    return lines


def _format_tb_validation_block(final_parsed: dict, tb_ev: dict) -> list[str]:
    """Render Layer-7 TB/card validation."""
    technical_tb = final_parsed.get("technical_tb_sheet")
    card_sheet = final_parsed.get("is_card_sheet")
    relationship = final_parsed.get("relationship", {}) or {}

    lines = ["## Layer-7 TB / Card-Sheet Validation", ""]
    lines.append(f"- Final TB/card sheet: `{card_sheet}`")
    lines.append(f"- Technical TB sheet: `{technical_tb}`")
    lines.append(f"- Path valid: `{relationship.get('path_valid', False)}`")

    path = relationship.get("main_to_tb_path", []) or []
    if path:
        lines.append(f"- Main-to-TB path: `{ ' → '.join(path) }`")
    else:
        lines.append("- Main-to-TB path: `none recorded`")

    lines.append("")

    if tb_ev:
        lines.append("### Why this TB sheet was chosen")
        lines.append("")
        lines += _format_nn_evidence(tb_ev)
        lines.append("")
    else:
        lines += ["*No explicit TB NN evidence was available for the chosen TB sheet.*", ""]

    return lines


def _signal_reason_lines(ev: dict) -> list[str]:
    if not ev:
        return ["- No signal evidence was available for the selected sheet."]

    lines: list[str] = []

    if ev.get("COA_SIGNAL") == 1:
        lines.append("- `COA_SIGNAL` fired because the sheet looked like a financial statement structure rather than a raw ledger.")
    else:
        lines.append("- `COA_SIGNAL` did not fire, so the sheet did not strongly present financial-statement section structure.")

    if ev.get("FORMULA_SIGNAL") == 1:
        lines.append("- `FORMULA_SIGNAL` fired because the sheet contains formulas, meaning it is computed rather than purely static.")
    else:
        lines.append("- `FORMULA_SIGNAL` did not fire, so the sheet did not show calculation-based aggregation evidence.")

    if ev.get("CROSS_REF_SIGNAL") == 1:
        lines.append("- `CROSS_REF_SIGNAL` fired because the sheet references other sheets, which supports a synthesis/output role.")
    else:
        lines.append("- `CROSS_REF_SIGNAL` did not fire, so the sheet did not show outward workbook dependency evidence.")

    if ev.get("REFERENCED_BY_SIGNAL") == 1:
        lines.append("- `REFERENCED_BY_SIGNAL` fired, meaning other sheets depend on this sheet.")
    else:
        lines.append("- `REFERENCED_BY_SIGNAL` did not fire, so the sheet was not strongly used as an upstream dependency.")

    if ev.get("COMPANY_COLUMN_SIGNAL") == 1:
        lines.append("- `COMPANY_COLUMN_SIGNAL` fired because company/entity-style columns were detected.")
    else:
        lines.append("- `COMPANY_COLUMN_SIGNAL` did not fire, which weakens the case for the sheet being a standard FS output.")

    if ev.get("AJE_SIGNAL") == 1:
        lines.append("- `AJE_SIGNAL` fired, which suggests adjustment-related content was present.")
    else:
        lines.append("- `AJE_SIGNAL` did not fire, so there was no strong adjustment-sheet indicator.")

    if ev.get("CONSOLIDATE_SIGNAL") == 1:
        lines.append("- `CONSOLIDATE_SIGNAL` fired, which supports a consolidation-oriented interpretation.")
    else:
        lines.append("- `CONSOLIDATE_SIGNAL` did not fire, so consolidation wording was not a major factor.")

    if ev.get("HAS_CODE_COLUMN") == 1:
        lines.append("- `HAS_CODE_COLUMN` fired, which is typical of source/TB sheets and works against final-FS classification.")
    else:
        lines.append("- `HAS_CODE_COLUMN` did not fire, which supports the sheet not being a raw code-ledger source.")

    if ev.get("HAS_DESCRIPTION_COLUMN") == 1:
        lines.append("- `HAS_DESCRIPTION_COLUMN` fired, which supports account-level TB/card-sheet structure.")
    else:
        lines.append("- `HAS_DESCRIPTION_COLUMN` did not fire, so account-description structure was not strongly detected.")

    if ev.get("HAS_FINAL_COLUMN") == 1:
        lines.append("- `HAS_FINAL_COLUMN` fired, which supports an ending-balance / TB amount-column interpretation.")
    else:
        lines.append("- `HAS_FINAL_COLUMN` did not fire, so there was no strong final amount-column signature.")

    if ev.get("FINAL_REFERENCE_SIGNAL") == 1:
        lines.append("- `FINAL_REFERENCE_SIGNAL` fired, meaning the final amount column appears to be referenced by upstream formulas.")
    else:
        lines.append("- `FINAL_REFERENCE_SIGNAL` did not fire, so upstream formula targeting was not a major factor.")

    if ev.get("TB_REFERENCE_SIGNAL") == 1:
        lines.append("- `TB_REFERENCE_SIGNAL` fired, which supports the sheet being used as a source/TB target.")
    else:
        lines.append("- `TB_REFERENCE_SIGNAL` did not fire, so TB-source usage was not explicitly confirmed.")

    if ev.get("STAGING_ROLE_SIGNAL") == 1:
        lines.append("- `STAGING_ROLE_SIGNAL` fired, which suggests bridge / AJE / support-sheet behavior.")
    else:
        lines.append("- `STAGING_ROLE_SIGNAL` did not fire, so staging-style wording was not a major factor.")

    if ev.get("HIDDEN_SIGNAL") == 1:
        lines.append("- `HIDDEN_SIGNAL` fired, which is a direct blocker for main-sheet candidacy.")
    else:
        lines.append("- `HIDDEN_SIGNAL` did not fire, so visibility was not a blocker.")

    return lines


def _layer_explanation_block(
    chosen: str | None,
    chosen_ev: dict,
    final_parsed: dict,
) -> list[str]:
    lines: list[str] = ["## Layer-by-layer explanation", ""]

    if not chosen:
        lines += [
            "No final sheet was selected, so the layer-by-layer explanation is limited to the final blocked state.",
            "",
        ]
        return lines

    lines += [
        f"The explanations below are generated **after the full pipeline completed**, using the final verified state for **`{chosen}`**.",
        "",
        "### L0 — Raw extraction",
        (
            "L0 gathered workbook-level evidence first: sheet names, headers, values, formulas, "
            "cross-sheet references, visibility, candidate code/description/final columns, and active-sheet context. "
            "This layer does not decide; it only collects raw workbook facts for later layers."
        ),
        "",
        "### L1 — Binary signals",
        "L1 converted raw workbook evidence into binary signals that the later layers can reason over.",
    ]
    lines += _signal_reason_lines(chosen_ev)
    lines += [""]

    l2_expl = []
    if chosen_ev.get("FS_PATTERN") == 1:
        l2_expl.append(
            "L2 chose `FS_PATTERN` because the selected sheet matched the full financial-statement signature "
            "(statement-like structure + formula behavior + cross-sheet references + company/entity evidence "
            "+ no source-sheet conflict)."
        )
    if chosen_ev.get("TB_PATTERN") == 1:
        l2_expl.append(
            "L2 chose `TB_PATTERN`, meaning the sheet behaved like a trial balance / source ledger rather than a final output."
        )
    if chosen_ev.get("STRONG_TB_PATTERN") == 1:
        l2_expl.append(
            "L2 also chose `STRONG_TB_PATTERN`, meaning the TB/source interpretation was reinforced by stronger structural or reference evidence."
        )
    if chosen_ev.get("PARTIAL_FS_PATTERN") == 1:
        l2_expl.append(
            "L2 chose `PARTIAL_FS_PATTERN`, meaning the sheet showed incomplete but meaningful FS evidence and needed deeper graph validation."
        )
    if chosen_ev.get("STAGING_PATTERN") == 1:
        l2_expl.append(
            "L2 chose `STAGING_PATTERN`, meaning the sheet looked like an AJE / bridge / support layer rather than the final report."
        )
    if not l2_expl:
        l2_expl.append(
            "L2 did not find a strong composite pattern, so later layers had to rely more on graph position, gates, and arbitration."
        )

    lines += ["### L2 — Pattern logic"]
    lines += l2_expl
    lines += [""]

    role = chosen_ev.get("role_in_graph", "UNKNOWN")
    outgoing = chosen_ev.get("outgoing_refs", []) or []
    incoming = chosen_ev.get("incoming_refs", []) or []
    l3_parts = [f"L3 assigned the sheet graph role **`{role}`**."]
    if outgoing:
        l3_parts.append(
            f"It has outgoing references to `{', '.join(outgoing)}`, which supports a computed/output role."
        )
    else:
        l3_parts.append("It has no outgoing references recorded in the final evidence.")
    if incoming:
        l3_parts.append(
            f"It also has incoming references from `{', '.join(incoming)}`."
        )
    else:
        l3_parts.append("It has no incoming references recorded in the final evidence.")
    if chosen_ev.get("consolidate", False):
        l3_parts.append("L3 also marked it as a consolidation-related sheet.")
    if chosen_ev.get("attention_boost", False):
        l3_parts.append("The sheet also received an attention boost from the graph layer.")
    if chosen_ev.get("aje_source_role", False):
        l3_parts.append("It was also flagged as behaving like an AJE/support source in the graph.")
    if chosen_ev.get("path_valid", False):
        pt = chosen_ev.get("path_to_tb", []) or []
        if pt:
            l3_parts.append(f"A valid path to TB was recorded as `{ ' → '.join(pt) }`.")

    lines += ["### L3 — Context graph", " ".join(l3_parts), ""]

    if chosen_ev.get("gate_passed"):
        l4_text = (
            "L4 allowed the sheet to continue because no final blocking gate fired "
            "for the chosen result in the final state."
        )
    else:
        l4_text = (
            f"L4 blocked the sheet with **`{chosen_ev.get('blocked_by')}`**. "
            "If the sheet still survived in the final answer, that could only happen if "
            "a later interpretation treated the issue as technical rather than critical."
        )

    lines += ["### L4 — Hard gates", l4_text, ""]

    conf = chosen_ev.get("confidence")
    technical_main = final_parsed.get("technical_main_sheet")
    if conf is not None:
        l5_text = (
            f"L5 assigned technical confidence **`{conf:.3f}`**. "
            f"The technical winner after softmax was **`{technical_main}`**."
        )
    else:
        l5_text = (
            f"L5 produced the technical winner **`{technical_main}`**, but no final confidence value was available in the extracted evidence."
        )

    lines += ["### L5 — Technical confidence / softmax", l5_text, ""]

    ba = final_parsed.get("business_arbitration", {}) or {}
    override_applied = ba.get("override_applied", False)
    technical_type = ba.get("technical_winner_sheet_type")
    presentation_candidate = ba.get("presentation_candidate") or ba.get("business_candidate")
    presentation_type = ba.get("presentation_candidate_sheet_type") or ba.get("business_candidate_sheet_type")
    decision_mode = final_parsed.get("decision_mode", "unknown")

    if override_applied:
        l6_text = (
            f"L6 overrode the technical winner because the technical sheet was classified as "
            f"**`{technical_type}`**, while **`{presentation_candidate}`** was classified as "
            f"**`{presentation_type}`** and judged to be the human-facing final report. "
            f"The final decision mode was **`{decision_mode}`**."
        )
    else:
        l6_text = (
            f"L6 did not override the technical result. The final decision mode was **`{decision_mode}`**, "
            f"so the system concluded that the technical winner was also the correct business answer "
            f"or that no safe presentation-sheet override existed."
        )

    lines += ["### L6 — Business arbitration", l6_text, ""]

    relationship = final_parsed.get("relationship", {}) or {}
    technical_tb = final_parsed.get("technical_tb_sheet")
    card_sheet = final_parsed.get("is_card_sheet")
    tb_path = relationship.get("main_to_tb_path", []) or []
    path_valid = relationship.get("path_valid", False)

    if card_sheet or technical_tb:
        if path_valid and tb_path:
            l7_text = (
                f"L7 selected **`{card_sheet}`** as the final TB/card sheet "
                f"(technical TB: **`{technical_tb}`**) and validated the main-to-TB relationship "
                f"through the path `{ ' → '.join(tb_path) }`."
            )
        else:
            l7_text = (
                f"L7 selected **`{card_sheet}`** as the final TB/card sheet "
                f"(technical TB: **`{technical_tb}`**), but the relationship path was not fully validated."
            )
    else:
        l7_text = "L7 did not identify a final TB/card sheet in the exported final state."

    lines += ["### L7 — TB / card-sheet validation", l7_text, ""]

    return lines


def _process_explanation_block(state, final_parsed: dict) -> list[str]:
    sessions = ensure_debug_trace(state).get("court_sessions", [])
    steps = ensure_debug_trace(state).get("steps", [])
    decision_mode = final_parsed.get("decision_mode", "unknown")
    chosen = state.get("main_sheet_name")
    technical_main = final_parsed.get("technical_main_sheet")
    presentation_main = final_parsed.get("presentation_main_sheet")
    technical_tb = final_parsed.get("technical_tb_sheet")
    card_sheet = final_parsed.get("is_card_sheet")
    relationship = final_parsed.get("relationship", {}) or {}

    lines = [
        "## Process explanation",
        "",
        "This explanation is written **after the entire pipeline finished**, not during execution.",
        "It summarizes how the system moved from workbook inspection to a final verified answer.",
        "",
        "### End-to-end process",
        "1. The workbook was inspected and a detector candidate was collected as a non-binding hint.",
        "2. The plan node translated the initial evidence into targeted checks for the research agent.",
        "3. The research agent used workbook tools to produce NN-layer evidence.",
        "4. The court stage reviewed the agent output and either approved it or forced revision.",
        "5. The synthesis node aggregated the evidence into a final technical decision.",
        "6. Python guardrails verified that critical blocks were respected.",
        "7. Layer-6 business arbitration checked whether the technical winner was also the human-facing final output.",
        "8. Layer-7 validated the TB/card sheet and its relationship to the final main sheet.",
        "9. Only after all of that completed was this markdown report written.",
        "",
        "### Final process outcome",
        f"- Final main sheet: `{chosen}`",
        f"- Technical main sheet: `{technical_main}`",
        f"- Presentation main sheet: `{presentation_main}`",
        f"- Final TB/card sheet: `{card_sheet}`",
        f"- Technical TB sheet: `{technical_tb}`",
        f"- Relationship path valid: `{relationship.get('path_valid', False)}`",
        f"- Decision mode: `{decision_mode}`",
        f"- Total logged pipeline steps: `{len(steps)}`",
        f"- Total court sessions: `{len(sessions)}`",
        "",
    ]
    return lines


def _build_decision_markdown(state) -> str:
    result = state.get("main_sheet_result", {})
    chosen = state.get("main_sheet_name")
    has_main = state.get("has_main_sheet", False)
    detector_cand = state.get("detector_candidate")
    active_sheet = result.get("active_sheet")

    final_answer_raw = state.get("final_answer", "")
    final_parsed = _parse_json_from_text(final_answer_raw) or {}

    technical_main = final_parsed.get("technical_main_sheet")
    presentation_main = final_parsed.get("presentation_main_sheet")
    technical_tb = final_parsed.get("technical_tb_sheet")
    card_sheet = final_parsed.get("is_card_sheet")
    relationship = final_parsed.get("relationship", {}) or {}
    decision_mode = final_parsed.get("decision_mode", "unknown")
    ba_block = final_parsed.get("business_arbitration", {})

    out_cands = result.get("output_candidates", [])
    src_cands = result.get("source_candidates", [])
    agent_result = (state.get("task_results") or [{}])[0].get("result", "")

    chosen_ev = _extract_nn_evidence(state, chosen)
    tb_ev = _extract_nn_evidence(state, technical_tb or card_sheet)

    lines = [
        "# Main Sheet Decision Report", "",
        f"- Run time: `{utc_now()}`",
        f"- Workbook: `{state.get('user_input')}`",
        f"- **Main sheet found: `{has_main}`**",
        f"- **Final main sheet: `{chosen}`**  ← synthesis + L6 authoritative",
        f"- **Technical main sheet: `{technical_main}`**  ← L5 softmax winner",
        f"- **Presentation main sheet: `{presentation_main}`**  ← L6 presentation candidate",
        f"- **Final TB/card sheet: `{card_sheet}`**  ← L7 validated source sheet",
        f"- **Technical TB sheet: `{technical_tb}`**  ← L7 structural TB winner",
        f"- **Decision mode: `{decision_mode}`**",
        f"- **Relationship path valid: `{relationship.get('path_valid', False)}`**",
        f"- Detector candidate (hint only): `{detector_cand}`",
        f"- Workbook active sheet: `{active_sheet}`",
        "",
    ]

    tb_path = relationship.get("main_to_tb_path", []) or []
    if tb_path:
        lines.append(f"- Main-to-TB path: `{ ' → '.join(tb_path) }`")
        lines.append("")

    if decision_mode == "business_override":
        lines += [
            "🔄 **Business Override Active**: Layer-6 arbitration detected a mismatch "
            "between the technical winner (computational hub) and the business-correct "
            "final presentation sheet. The REPORTING_FS presentation sheet was promoted.",
            "",
        ]
    elif decision_mode == "business_override_with_tb_validation":
        lines += [
            "✅🔄 **Business Override + TB Validation Active**: Layer-6 promoted the presentation sheet, "
            "and Layer-7 also validated a main-to-TB relationship for the final result.",
            "",
        ]
    elif decision_mode == "technical_default":
        lines += [
            "✅ **Technical Default**: Layer-6 confirmed that the technical winner "
            "is also the business-correct final sheet. No override was needed.",
            "",
        ]

    if detector_cand and detector_cand != chosen:
        lines += [
            "⚠️ **Detector override**: the heuristic detector suggested "
            f"`{detector_cand}` but the NN PROMAT + synthesis chose `{chosen}`.",
            "This is correct — the detector is a hint generator, not a decision maker.",
            "",
        ]

    lines += _process_explanation_block(state, final_parsed)

    sessions = ensure_debug_trace(state).get("court_sessions", [])
    if sessions:
        lines += ["## Court Sessions", ""]
        for cs in sessions:
            lines.append(
                f"- `{cs['agent_under_review']}` | "
                f"Attempt {cs['attempt']} | **{cs['verdict']}**"
            )
        lines.append("")

    lines += ["## Why this sheet was chosen", ""]
    if chosen:
        lines.append(
            f"Synthesis + Layer-6 selected **`{chosen}`** based on NN PROMAT layer evidence:"
        )
        lines += _format_nn_evidence(chosen_ev)
    else:
        lines.append(
            "No main sheet identified — all candidates were blocked by NN gates or Python guardrails."
        )
        if detector_cand:
            det_ev = _extract_nn_evidence(state, detector_cand)
            lines.append(
                f"The detector suggested `{detector_cand}` but it failed gate checks: "
                f"{det_ev.get('blocked_by', 'unknown')}."
            )
    lines.append("")

    lines += _layer_explanation_block(chosen, chosen_ev, final_parsed)

    lines += ["## Layer-6 Business Arbitration", ""]
    lines += _format_business_arbitration_block(ba_block)
    lines.append("")

    lines += _format_tb_validation_block(final_parsed, tb_ev)

    lines += [
        "## Decision authority chain", "",
        "```",
        "L7 TB / Card-Sheet Validation",
        "      ↑",
        "L6 Business Arbitration (presentation bridge)",
        "      ↑",
        "L5 Softmax (technical winner)",
        "      ↑",
        "L4 Hard gates (GATE_1–GATE_5)",
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

    if agent_result:
        lines += [
            "## Research agent NN output", "",
            "```json", str(agent_result), "```", "",
        ]

    lines += ["## Candidate rankings (heuristic — for reference only)", ""]
    lines.append(_format_candidate_block(out_cands, "Top output candidates"))
    lines.append(_format_candidate_block(src_cands, "Top source candidates"))

    if state.get("final_answer"):
        lines += ["## Final synthesis JSON", "", "```json", str(state["final_answer"]), "```", ""]

    return "\n".join(lines).strip() + "\n"


def export_artifacts(state) -> tuple[str, str]:
    trace = ensure_debug_trace(state)
    trace["finished_at"] = utc_now()
    trace["status"] = "success"

    export_dir = Path("debug_traces")
    export_dir.mkdir(parents=True, exist_ok=True)
    run_id = trace["run_id"]

    payload = {
        "run_id": run_id,
        "started_at": trace["started_at"],
        "finished_at": trace["finished_at"],
        "status": trace["status"],
        "input": {"file_path": state.get("user_input")},
        "has_main_sheet": state.get("has_main_sheet", False),
        "main_sheet_name": state.get("main_sheet_name"),
        "technical_main_sheet": state.get("technical_main_sheet"),
        "presentation_main_sheet": state.get("presentation_main_sheet"),
        "is_card_sheet": state.get("is_card_sheet"),
        "technical_tb_sheet": state.get("technical_tb_sheet"),
        "decision_mode": state.get("decision_mode"),
        "relationship": state.get("relationship"),
        "detector_candidate": state.get("detector_candidate"),
        "excel_summary": state.get("excel_summary"),
        "main_sheet_result": state.get("main_sheet_result"),
        "analysis": state.get("analysis"),
        "plan": state.get("plan"),
        "task_results": state.get("task_results"),
        "court_sessions": trace.get("court_sessions", []),
        "final_answer": state.get("final_answer"),
        "steps": trace.get("steps", []),
        "agents": trace.get("agents", []),
        "errors": trace.get("errors", []),
    }

    json_file = export_dir / f"{run_id}.json"
    md_file = export_dir / f"{run_id}.md"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(_build_decision_markdown(state))

    state["export_file"] = state["debug_trace_file"] = str(json_file)
    state["md_export_file"] = str(md_file)
    return str(json_file), str(md_file)