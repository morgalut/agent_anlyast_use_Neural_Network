"""
graph.py — ORC LangGraph pipeline  (NN PROMAT edition)

═══════════════════════════════════════════════════════════════════════
  DEEP ANALYSIS — what was wrong in the previous version
═══════════════════════════════════════════════════════════════════════

BUG A — Guardrails read the OLD evidence schema (critical mismatch)
────────────────────────────────────────────────────────────────────
  Old schema field:  evidence.company_columns_confirmed  (bool)
  Old schema field:  evidence.is_hidden                  (bool)
  Old schema field:  confidence_score                    (int 0-100)

  New NN schema:     nn_evidence.layer1.COMPANY_COLUMN_SIGNAL  (0/1)
  New NN schema:     nn_evidence.layer1.HIDDEN_SIGNAL          (0/1)
  New NN schema:     nn_evidence.layer4.passed                 (bool)
  New NN schema:     confidence                                (float 0.0-1.0)

  Effect: _apply_promat_guardrails always saw company_confirmed=False
  because the key it was looking for no longer existed in the output.
  This means GATE_2 was never truly enforced in Python — the LLM's NN
  reasoning was the only defence, which is not enough.

BUG B — runner_up confidence divided by 100 (wrong scale)
───────────────────────────────────────────────────────────
  Old code:  ru_conf = float(ru_ev.get("confidence_score", 0)) / 100.0
  The new NN schema reports confidence as a float in [0.0, 1.0].
  Dividing by 100 turned 0.75 into 0.0075 → runner_up NEVER promoted.

BUG C — _build_decision_markdown reads output_score / score_breakdown
──────────────────────────────────────────────────────────────────────
  The markdown export still showed "Output score: 23" from the old
  heuristic scoring system.  The NN system produces no such field.
  The markdown should show NN layer evidence instead.

BUG D — Guardrail reasoning still written in Hebrew
─────────────────────────────────────────────────────
  After switching all prompts to English, the guardrail update block
  still wrote "reasoning_he" with Hebrew text.  Inconsistent with the
  rest of the system, and the synthesis schema now uses "reasoning".

BUG E — No GATE-level guardrail
─────────────────────────────────
  The Python guardrails checked company_columns and is_hidden but never
  checked the NN's own layer4.passed flag.  A sheet the NN explicitly
  blocked (e.g. via GATE_3 — TB sheet) could still slip through if the
  LLM returned it as main_sheet_name while the guardrail code found no
  evidence to the contrary in the old schema fields.

═══════════════════════════════════════════════════════════════════════
  FIXES IN THIS VERSION
═══════════════════════════════════════════════════════════════════════

  • _extract_nn_evidence()        — reads new NN schema fields correctly
  • _apply_nn_guardrails()        — enforces all 4 NN gates in Python
  • runner_up confidence          — no longer divided by 100
  • _build_decision_markdown()    — shows NN layer signals, not old scores
  • all guardrail reasoning       — English, matches new schema
  • synthesis owns the decision   — unchanged, still authoritative
"""

from __future__ import annotations

import json
import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph

from app.config.config import get_settings
from app.model.state import OrchestratorState
from app.server.agent.agents import (
    build_defense_agent,
    build_judge_agent,
    build_plaintiff_agent,
    build_research_agent,
)
from app.server.orc.promat import (
    build_analyze_prompt,
    build_plan_prompt,
    build_research_task_instruction,
    build_synthesize_prompt,
)
from app.server.orc.promat.court_prompt import (
    build_agent_revision_prompt,
    build_court_user_prompt,
)
from app.tools.tools import detect_main_sheet, inspect_workbook

MAX_COURT_RETRIES = 2


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_llm() -> AzureChatOpenAI:
    s = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=s.azure_endpoint,
        api_version=s.azure_api_version,
        deployment_name=s.azure_deployment,
        api_key=s.azure_api_key,
        temperature=0,
    )


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


def ensure_debug_trace(state: OrchestratorState) -> Dict[str, Any]:
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


def add_step_log(state, node: str, data: Dict[str, Any]) -> None:
    ensure_debug_trace(state)["steps"].append(
        {"timestamp": utc_now(), "node": node, "data": safe_preview(data)}
    )


def add_agent_log(state, task_id, agent_name, instruction, response=None, error=None):
    ensure_debug_trace(state)["agents"].append({
        "timestamp":   utc_now(),
        "task_id":     task_id,
        "agent":       agent_name,
        "instruction": instruction,
        "response":    safe_preview(response),
        "error":       error,
    })


def add_court_log(state, agent_name, attempt, p_out, d_out, j_out, verdict):
    ensure_debug_trace(state)["court_sessions"].append({
        "timestamp":          utc_now(),
        "agent_under_review": agent_name,
        "attempt":            attempt,
        "plaintiff":          safe_preview(p_out),
        "defense":            safe_preview(d_out),
        "judge":              safe_preview(j_out),
        "verdict":            verdict,
    })


def add_error_log(state, where: str, error: Exception) -> None:
    ensure_debug_trace(state)["errors"].append({
        "timestamp": utc_now(),
        "where":     where,
        "type":      type(error).__name__,
        "message":   str(error),
        "traceback": traceback.format_exc(),
    })


def _find_profile(state, sheet_name: str | None) -> dict | None:
    if not sheet_name:
        return None
    for p in state.get("main_sheet_result", {}).get("profiles", []):
        if p.get("sheet_name") == sheet_name:
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _parse_synthesis_result(raw: str) -> tuple[str | None, bool, float, dict]:
    """
    Parse synthesize_node LLM output.
    Returns (main_sheet_name, main_sheet_exists, confidence, full_parsed_dict).
    Reads from api_response first (the NN PROMAT canonical field).
    confidence is expected to be a float in [0.0, 1.0].
    """
    parsed = _parse_json_from_text(raw)
    if not parsed:
        return None, False, 0.0, {}

    api    = parsed.get("api_response", {})
    name   = api.get("main_sheet_name") or parsed.get("main_sheet_name")
    exists = api.get("main_sheet_exists", parsed.get("main_sheet_exists", False))
    conf   = float(parsed.get("confidence", 0.0))

    # Normalise: exists must be False when name is null
    if not name:
        exists, conf = False, 0.0

    return name, bool(exists), conf, parsed


# ─────────────────────────────────────────────────────────────────────────────
#  NN evidence extraction  (FIX A — reads new nn_evidence schema)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  NN-aware guardrails  (FIX A + B + D + E)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_nn_guardrails(parsed: dict, state) -> dict:
    """
    Enforce NN PROMAT hard gates in Python after synthesis.

    This is a deterministic safety net — it catches cases where the LLM
    produced a syntactically valid JSON answer that violates a gate rule.

    Gates mirrored from nn_promat_core.py Layer 4:
      G1 — HIDDEN_SIGNAL = 1          → block (mirrors GATE_1)
      G2 — COMPANY_COLUMN_SIGNAL = 0  → block unless CONSOLIDATE exempt
                                         (mirrors GATE_2)
      G3 — TB_PATTERN = 1             → block (mirrors GATE_3)
      G4 — layer4.passed = false      → block (any gate fired in NN)
      G5 — confidence < 0.40          → not confident enough

    Runner-up promotion:
      If the winner is blocked, attempt to promote the runner_up.
      FIX B: runner_up confidence is read directly as float 0-1 —
             no /100 division.

    All reasoning fields use English to match the new schema.
    """
    result     = dict(parsed)
    sheet_name = result.get("main_sheet_name")
    confidence = float(result.get("confidence", 0.0))

    # Pull NN evidence for the proposed sheet
    ev = _extract_nn_evidence(state, sheet_name)

    # Also check what the synthesis node reported in nn_synthesis
    nn_syn = result.get("nn_synthesis", {})

    disqualify_reason: str | None = None

    if result.get("main_sheet_exists") and sheet_name:

        # G4 — layer4.passed is the NN's own verdict (catch-all gate check)
        if ev and not ev.get("gate_passed", True):
            blocked_by = ev.get("blocked_by", "unknown gate")
            disqualify_reason = (
                f"G4: NN layer4.passed=false (blocked_by={blocked_by}) "
                f"— the NN itself disqualified this sheet"
            )

        # G1 — hidden sheet
        elif ev.get("HIDDEN_SIGNAL") == 1:
            disqualify_reason = "G1: HIDDEN_SIGNAL=1 — hidden sheets cannot be main sheet"

        # G2 — no company columns (GATE_2 equivalent)
        elif (
            ev.get("COMPANY_COLUMN_SIGNAL") == 0
            and ev.get("CONSOLIDATE_SIGNAL") == 0
            and not ev.get("consolidate", False)
        ):
            disqualify_reason = (
                "G2: COMPANY_COLUMN_SIGNAL=0 and not a CONSOLIDATE sheet "
                "— no company columns means this cannot be the FS main sheet "
                "(this is the CF-type error guard)"
            )

        # G3 — TB pattern sheet
        elif ev.get("TB_PATTERN") == 1:
            disqualify_reason = "G3: TB_PATTERN=1 — trial balance / ledger sheets are sources, not main sheets"

        # G5 — low confidence
        elif confidence < 0.40:
            disqualify_reason = f"G5: confidence={confidence:.2f} is below 0.40 threshold"

    if not disqualify_reason:
        return result   # all gates passed

    add_step_log(state, "guardrail:disqualify", {
        "sheet":  sheet_name,
        "reason": disqualify_reason,
        "nn_evidence": safe_preview(ev, 500),
    })

    # ── Attempt runner_up promotion ───────────────────────────────────────────
    runner_up = result.get("runner_up")
    if runner_up:
        ru_ev   = _extract_nn_evidence(state, runner_up)
        # FIX B: confidence is already float 0-1 — no /100
        ru_conf = ru_ev.get("confidence", 0.0)

        ru_passes = (
            ru_ev.get("gate_passed", False)
            and ru_ev.get("COMPANY_COLUMN_SIGNAL", 0) == 1
            and ru_ev.get("HIDDEN_SIGNAL", 0) == 0
            and ru_ev.get("TB_PATTERN", 0) == 0
            and ru_conf >= 0.40
        )

        if ru_passes:
            add_step_log(state, "guardrail:runner_up_promoted", {
                "blocked": sheet_name,
                "promoted": runner_up,
                "ru_confidence": ru_conf,
            })
            result.update({
                "main_sheet_exists": True,
                "main_sheet_name":   runner_up,
                "confidence":        ru_conf,
                "reasoning":         (
                    f"Sheet '{sheet_name}' was blocked by Python guardrails "
                    f"({disqualify_reason}). Runner-up '{runner_up}' passed all "
                    f"NN gates and was promoted."
                ),
                "api_response": {
                    "main_sheet_exists": True,
                    "main_sheet_name":   runner_up,
                },
            })
            return result

    # ── No valid sheet found ──────────────────────────────────────────────────
    add_step_log(state, "guardrail:no_valid_sheet", {
        "reason":   disqualify_reason,
        "runner_up_attempted": runner_up,
        "runner_up_passed":    False,
    })
    result.update({
        "main_sheet_exists": False,
        "main_sheet_name":   None,
        "confidence":        0.0,
        "reasoning":         (
            f"No valid main sheet found. '{sheet_name}' was blocked: "
            f"{disqualify_reason}. Runner-up '{runner_up}' also failed "
            "NN gate checks."
        ),
        "api_response": {
            "main_sheet_exists": False,
            "main_sheet_name":   None,
        },
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Markdown export  (FIX C — shows NN signals, not old score fields)
# ─────────────────────────────────────────────────────────────────────────────

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
        "COA_SIGNAL": ev.get("COA_SIGNAL"),
        "COMPANY_COLUMN_SIGNAL": ev.get("COMPANY_COLUMN_SIGNAL"),
        "CROSS_REF_SIGNAL": ev.get("CROSS_REF_SIGNAL"),
        "FORMULA_SIGNAL": ev.get("FORMULA_SIGNAL"),
        "HIDDEN_SIGNAL": ev.get("HIDDEN_SIGNAL"),
        "AJE_SIGNAL": ev.get("AJE_SIGNAL"),
        "CONSOLIDATE_SIGNAL": ev.get("CONSOLIDATE_SIGNAL"),
        "CODE_COLUMN_SIGNAL": ev.get("CODE_COLUMN_SIGNAL"),
        "FINAL_COLUMN_SIGNAL": ev.get("FINAL_COLUMN_SIGNAL"),
    }
    fired   = [k for k, v in l1_signals.items() if v == 1]
    silent  = [k for k, v in l1_signals.items() if v == 0]
    if fired:
        lines.append(f"  - **L1 active signals**: `{'` `'.join(fired)}`")
    if silent:
        lines.append(f"  - L1 silent signals: `{'` `'.join(silent)}`")
    # Layer 2
    if ev.get("FS_PATTERN") is not None:
        pat = []
        if ev.get("FS_PATTERN"):   pat.append("FS_PATTERN ✓")
        if ev.get("TB_PATTERN"):   pat.append("TB_PATTERN ✓")
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


def _build_decision_markdown(state) -> str:
    result         = state.get("main_sheet_result", {})
    chosen         = state.get("main_sheet_name")
    has_main       = state.get("has_main_sheet", False)
    detector_cand  = state.get("detector_candidate")
    active_sheet   = result.get("active_sheet")
    main_source    = state.get("final_answer") and _parse_json_from_text(
                         state.get("final_answer", "")
                     ) or {}
    main_source_nm = main_source.get("main_source_sheet") if isinstance(main_source, dict) else None
    out_cands      = result.get("output_candidates", [])
    src_cands      = result.get("source_candidates", [])
    agent_result   = (state.get("task_results") or [{}])[0].get("result", "")

    # Pull NN evidence for the chosen sheet
    chosen_ev = _extract_nn_evidence(state, chosen)

    lines = [
        "# Main Sheet Decision Report", "",
        f"- Run time: `{utc_now()}`",
        f"- Workbook: `{state.get('user_input')}`",
        f"- **Main sheet found: `{has_main}`**",
        f"- **Chosen main sheet: `{chosen}`**  ← synthesis-authoritative",
        f"- Detector candidate (hint only): `{detector_cand}`",
        f"- Strongest source sheet: `{main_source_nm}`",
        f"- Workbook active sheet: `{active_sheet}`",
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
            f"Synthesis selected **`{chosen}`** based on NN PROMAT layer evidence:"
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

    # Authority chain
    lines += [
        "## Decision authority chain", "",
        "```",
        "Synthesis + Python guardrails",
        "      ↑",
        "Court (plaintiff / defense / judge)",
        "      ↑",
        "Research Agent (NN PROMAT layers 0-5)",
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


# ─────────────────────────────────────────────────────────────────────────────
#  Court helpers
# ─────────────────────────────────────────────────────────────────────────────

def _invoke_agent(agent, msg: str) -> str:
    return agent.invoke(
        {"messages": [{"role": "user", "content": msg}]}
    )["messages"][-1].content


def _parse_judge_verdict(j_out: str) -> str:
    for pat in [r'"verdict"\s*:\s*"([^"]+)"', r"'verdict'\s*:\s*'([^']+)'"]:
        m = re.search(pat, j_out)
        if m:
            return m.group(1)
    return "approved"


def run_court_session(state, agent_name: str, agent_output: str, attempt: int):
    p = build_plaintiff_agent()
    d = build_defense_agent()
    j = build_judge_agent()

    pp = build_court_user_prompt(agent_name=agent_name, agent_output=agent_output)
    po = _invoke_agent(p, pp)
    add_agent_log(state, f"court_p_{attempt}", "plaintiff_agent", pp, po)

    dp = build_court_user_prompt(
        agent_name=agent_name, agent_output=agent_output, plaintiff_charges=po
    )
    do = _invoke_agent(d, dp)
    add_agent_log(state, f"court_d_{attempt}", "defense_agent", dp, do)

    jp = build_court_user_prompt(
        agent_name=agent_name, agent_output=agent_output,
        plaintiff_charges=po, defense_arguments=do,
    )
    jo = _invoke_agent(j, jp)
    add_agent_log(state, f"court_j_{attempt}", "judge_agent", jp, jo)

    verdict = _parse_judge_verdict(jo)
    add_court_log(state, agent_name, attempt, po, do, jo, verdict)
    return verdict, jo


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline nodes
# ─────────────────────────────────────────────────────────────────────────────

def analyze_node(state: OrchestratorState) -> OrchestratorState:
    """
    Detector result stored as detector_candidate (hint only).
    main_sheet_name left None — synthesis will set it authoritatively.
    """
    ensure_debug_trace(state)
    try:
        llm       = get_llm()
        file_path = state["user_input"]
        add_step_log(state, "analyze:start", {"file_path": file_path})

        excel_summary     = inspect_workbook(file_path)
        main_sheet_result = detect_main_sheet(file_path)
        detector_cand     = main_sheet_result.get("main_sheet")  # hint only

        prompt = build_analyze_prompt(
            excel_summary=excel_summary,
            main_sheet_result=main_sheet_result,
        )
        result = llm.invoke(prompt)

        add_step_log(state, "analyze:end", {"detector_candidate": detector_cand})

        return {
            **state,
            "excel_summary":      excel_summary,
            "main_sheet_result":  main_sheet_result,
            "detector_candidate": detector_cand,   # hint only — never sticky
            "main_sheet_name":    None,             # not committed
            "has_main_sheet":     False,            # not committed
            "analysis":           result.content,
            "next_step":          "plan",
        }
    except Exception as e:
        add_error_log(state, "analyze_node", e)
        state["error"] = str(e)
        raise


def plan_node(state: OrchestratorState) -> OrchestratorState:
    try:
        llm = get_llm()
        add_step_log(state, "plan:start", {
            "detector_candidate": state.get("detector_candidate")
        })
        prompt = build_plan_prompt(
            analysis=state.get("analysis", ""),
            main_sheet_result=state.get("main_sheet_result", {}),
        )
        result = llm.invoke(prompt)
        task_instruction = build_research_task_instruction(
            file_path=state["user_input"], plan=result.content
        )
        tasks = [{"id": "task_1", "agent": "research_agent", "instruction": task_instruction}]
        add_step_log(state, "plan:end", {"plan": result.content})
        return {**state, "plan": result.content, "tasks": tasks, "next_step": "act"}
    except Exception as e:
        add_error_log(state, "plan_node", e)
        state["error"] = str(e)
        raise


def act_node(state: OrchestratorState) -> OrchestratorState:
    try:
        research_agent = build_research_agent()
        results: list[dict] = []
        add_step_log(state, "act:start", {"tasks": state.get("tasks", [])})

        for task in state.get("tasks", []):
            tid, aname, instr = task["id"], task["agent"], task["instruction"]
            if aname != "research_agent":
                msg = f"Unknown agent: {aname}"
                add_agent_log(state, tid, aname, instr, error=msg)
                results.append({"task_id": tid, "agent": aname, "result": msg})
                continue
            try:
                resp = research_agent.invoke(
                    {"messages": [{"role": "user", "content": instr}]}
                )
                out = resp["messages"][-1].content
                add_agent_log(state, tid, aname, instr, response=out)
                results.append({"task_id": tid, "agent": aname, "result": out})
            except Exception as ae:
                add_agent_log(state, tid, aname, instr, error=str(ae))
                add_error_log(state, f"act:{aname}", ae)
                results.append({
                    "task_id": tid, "agent": aname,
                    "result": f"Agent error: {ae}",
                })

        add_step_log(state, "act:end", {"count": len(results)})
        return {**state, "task_results": results, "next_step": "court"}
    except Exception as e:
        add_error_log(state, "act_node", e)
        state["error"] = str(e)
        raise


def court_node(state: OrchestratorState) -> OrchestratorState:
    try:
        add_step_log(state, "court:start", {"count": len(state.get("task_results", []))})
        reviewed: list[dict] = []

        for task in state.get("task_results", []):
            tid, aname, cur = task["task_id"], task["agent"], task["result"]
            meta:   list[dict] = []
            attempt            = 1
            final_v            = "approved"
            final_j            = ""

            while attempt <= MAX_COURT_RETRIES + 1:
                v, jo = run_court_session(state, aname, cur, attempt)
                final_v, final_j = v, jo
                meta.append({
                    "attempt":       attempt,
                    "verdict":       v,
                    "judge_summary": safe_preview(jo, 400),
                })

                if v in ("approved", "approved_with_note"):
                    break
                if attempt > MAX_COURT_RETRIES:
                    add_step_log(state, "court:max_retries", {
                        "task_id": tid, "verdict": v
                    })
                    break
                if v in ("revise_and_retry", "reject"):
                    try:
                        rp  = build_agent_revision_prompt(aname, cur, jo)
                        rr  = build_research_agent().invoke(
                            {"messages": [{"role": "user", "content": rp}]}
                        )
                        cur = rr["messages"][-1].content
                        add_agent_log(state, f"revision_{attempt}", aname, rp, cur)
                    except Exception as re_err:
                        add_error_log(state, f"court:revision_{attempt}", re_err)
                        break
                attempt += 1

            reviewed.append({
                "task_id":           tid,
                "agent":             aname,
                "result":            cur,
                "court_verdict":     final_v,
                "court_judge_output": final_j,
                "court_sessions":    meta,
            })

        add_step_log(state, "court:end", {"reviewed": len(reviewed)})
        return {**state, "task_results": reviewed, "next_step": "synthesize"}
    except Exception as e:
        add_error_log(state, "court_node", e)
        state["error"] = str(e)
        raise


def synthesize_node(state: OrchestratorState) -> OrchestratorState:
    """
    Final decision owner.

    1. Call synthesis LLM — produces NN-aggregated verdict.
    2. Parse the JSON response.
    3. Apply Python NN guardrails (deterministic safety net).
    4. Write authoritative main_sheet_name + has_main_sheet to state.

    The export node reads from state — always sees the verified answer.
    """
    try:
        llm = get_llm()
        add_step_log(state, "synthesize:start", {
            "detector_candidate": state.get("detector_candidate"),
            "task_count":         len(state.get("task_results", [])),
        })

        prompt = build_synthesize_prompt(
            analysis=state.get("analysis", ""),
            plan=state.get("plan", ""),
            main_sheet_result=state.get("main_sheet_result", {}),
            task_results=state.get("task_results", []),
        )
        llm_result  = llm.invoke(prompt)
        raw_content = llm_result.content

        # Step 1 — parse synthesis JSON
        name, exists, confidence, parsed = _parse_synthesis_result(raw_content)
        add_step_log(state, "synthesize:parsed", {
            "raw_name":       name,
            "raw_exists":     exists,
            "confidence":     confidence,
            "detector_was":   state.get("detector_candidate"),
        })

        # Step 2 — apply NN-aware Python guardrails
        if parsed:
            guarded     = _apply_nn_guardrails(parsed, state)
            name        = guarded.get("main_sheet_name")
            exists      = bool(guarded.get("main_sheet_exists", False))
            confidence  = float(guarded.get("confidence", 0.0))
            raw_content = json.dumps(guarded, ensure_ascii=False)

        add_step_log(state, "synthesize:end", {
            "final_name":  name,
            "final_exists": exists,
            "confidence":  confidence,
        })

        return {
            **state,
            "final_answer":    raw_content,
            "main_sheet_name": name,     # synthesis-owned, authoritative
            "has_main_sheet":  exists,   # synthesis-owned, authoritative
            "next_step":       "export",
        }
    except Exception as e:
        add_error_log(state, "synthesize_node", e)
        state["error"] = str(e)
        raise


def export_node(state: OrchestratorState) -> OrchestratorState:
    try:
        add_step_log(state, "export:start", {
            "main_sheet_name":    state.get("main_sheet_name"),
            "has_main_sheet":     state.get("has_main_sheet", False),
            "detector_candidate": state.get("detector_candidate"),
        })
        json_file, md_file = export_artifacts(state)
        add_step_log(state, "export:end", {"json": json_file, "md": md_file})
        return {
            **state,
            "export_file":     json_file,
            "md_export_file":  md_file,
            "next_step":       "done",
        }
    except Exception as e:
        add_error_log(state, "export_node", e)
        state["error"] = str(e)
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  Routing + graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def route_after_analyze(state):    return "plan"
def route_after_plan(state):       return "act"
def route_after_act(state):        return "court"
def route_after_court(state):      return "synthesize"
def route_after_synthesize(state): return "export"


def build_graph():
    g = StateGraph(OrchestratorState)
    g.add_node("analyze",    analyze_node)
    g.add_node("plan",       plan_node)
    g.add_node("act",        act_node)
    g.add_node("court",      court_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("export",     export_node)

    g.set_entry_point("analyze")
    g.add_conditional_edges("analyze",    route_after_analyze,    {"plan":       "plan"})
    g.add_conditional_edges("plan",       route_after_plan,       {"act":        "act"})
    g.add_conditional_edges("act",        route_after_act,        {"court":      "court"})
    g.add_conditional_edges("court",      route_after_court,      {"synthesize": "synthesize"})
    g.add_conditional_edges("synthesize", route_after_synthesize, {"export":     "export"})
    g.add_edge("export", END)
    return g.compile()