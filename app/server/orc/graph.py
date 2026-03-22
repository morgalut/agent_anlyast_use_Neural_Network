from __future__ import annotations

import json
import re
import traceback
import uuid
from datetime import datetime
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
    build_l6_plaintiff_agent,
    build_l6_defense_agent,
    build_l6_judge_agent,
)
from app.server.orc.md_export import export_artifacts
from app.server.orc.promat import (
    build_analyze_prompt,
    build_plan_prompt,
    build_research_task_instruction,
    build_synthesize_prompt,
)
from app.server.orc.promat.court_prompt import (
    build_agent_revision_prompt,
    build_court_user_prompt,
    build_l6_court_user_prompt,
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


def add_step_log(state, node: str, data: Dict[str, Any]) -> None:
    ensure_debug_trace(state)["steps"].append(
        {"timestamp": utc_now(), "node": node, "data": safe_preview(data)}
    )


def add_agent_log(state, task_id, agent_name, instruction, response=None, error=None):
    ensure_debug_trace(state)["agents"].append({
        "timestamp": utc_now(),
        "task_id": task_id,
        "agent": agent_name,
        "instruction": instruction,
        "response": safe_preview(response),
        "error": error,
    })


def add_court_log(state, agent_name, attempt, p_out, d_out, j_out, verdict):
    ensure_debug_trace(state)["court_sessions"].append({
        "timestamp": utc_now(),
        "agent_under_review": agent_name,
        "attempt": attempt,
        "plaintiff": safe_preview(p_out),
        "defense": safe_preview(d_out),
        "judge": safe_preview(j_out),
        "verdict": verdict,
    })


def add_error_log(state, where: str, error: Exception) -> None:
    ensure_debug_trace(state)["errors"].append({
        "timestamp": utc_now(),
        "where": where,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    })


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
    Reads from api_response first.
    """
    parsed = _parse_json_from_text(raw)
    if not parsed:
        return None, False, 0.0, {}

    api = parsed.get("api_response", {})
    name = api.get("main_sheet_name") or parsed.get("main_sheet_name")
    exists = api.get("main_sheet_exists", parsed.get("main_sheet_exists", False))
    conf = float(parsed.get("confidence", 0.0) or 0.0)

    if not name:
        exists, conf = False, 0.0

    return name, bool(exists), conf, parsed


def _clean_sheet_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


# ─────────────────────────────────────────────────────────────────────────────
#  Evidence indexing
# ─────────────────────────────────────────────────────────────────────────────

def _sheet_title_from_candidates(state, sheet_name: str | None) -> str:
    if not sheet_name:
        return ""
    msr = state.get("main_sheet_result", {})
    for bucket in ("output_candidates", "source_candidates"):
        for item in msr.get(bucket, []) or []:
            if item.get("sheet") == sheet_name:
                return str(item.get("title") or "")
    return ""


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def _coerce_int01(value: Any) -> int:
    try:
        return 1 if int(value) == 1 else 0
    except Exception:
        return 0


def _extract_candidate_sheet_names_from_state(state) -> list[str]:
    names: set[str] = set()

    msr = state.get("main_sheet_result", {})
    for bucket in ("output_candidates", "source_candidates", "profiles"):
        for item in msr.get(bucket, []) or []:
            sheet = item.get("sheet") or item.get("sheet_name")
            if sheet:
                names.add(str(sheet).strip())

    for field in ("detector_candidate", "main_sheet_name"):
        val = state.get(field)
        if val:
            names.add(str(val).strip())

    for task in state.get("task_results", []):
        parsed = _parse_json_from_text(task.get("result", ""))
        if not parsed:
            continue
        for field in (
            "main_sheet_name",
            "main_source_sheet_name",
            "technical_main_sheet",
            "business_main_sheet",
            "runner_up",
        ):
            val = parsed.get(field)
            if val:
                names.add(str(val).strip())

        # also collect names from structured evidence maps
        sheet_evidence = parsed.get("sheet_evidence", {}) or {}
        if isinstance(sheet_evidence, dict):
            for sheet in sheet_evidence.keys():
                if sheet:
                    names.add(str(sheet).strip())

        legacy_evidence = parsed.get("evidence", {}) or {}
        if isinstance(legacy_evidence, dict):
            for sheet, payload in legacy_evidence.items():
                if (
                    sheet
                    and isinstance(payload, dict)
                    and sheet not in {"ocr_used", "company_columns", "sections_found"}
                ):
                    names.add(str(sheet).strip())

    return sorted(n for n in names if n)


def _derive_sheet_evidence_from_single_task(parsed: dict, state) -> dict[str, dict]:
    evidence_map: dict[str, dict] = {}

    def _make_sheet_record(sheet_name: str, payload: dict) -> dict:
        l1 = payload.get("layer1", {}) or {}
        l2 = payload.get("layer2", {}) or {}
        l3 = payload.get("layer3", {}) or {}
        l4 = payload.get("layer4", {}) or {}

        return {
            "sheet_name": sheet_name,
            "evidence_status": "explicit",
            "source_task_type": "research_agent",
            "COMPANY_COLUMN_SIGNAL": _coerce_int01(l1.get("COMPANY_COLUMN_SIGNAL", 0)),
            "HIDDEN_SIGNAL": _coerce_int01(l1.get("HIDDEN_SIGNAL", 0)),
            "COA_SIGNAL": _coerce_int01(l1.get("COA_SIGNAL", 0)),
            "CONSOLIDATE_SIGNAL": _coerce_int01(l1.get("CONSOLIDATE_SIGNAL", 0)),
            "CROSS_REF_SIGNAL": _coerce_int01(l1.get("CROSS_REF_SIGNAL", 0)),
            "AJE_SIGNAL": _coerce_int01(l1.get("AJE_SIGNAL", 0)),
            "FORMULA_SIGNAL": _coerce_int01(l1.get("FORMULA_SIGNAL", 0)),
            "CODE_COLUMN_SIGNAL": _coerce_int01(l1.get("CODE_COLUMN_SIGNAL", 0)),
            "FINAL_COLUMN_SIGNAL": _coerce_int01(l1.get("FINAL_COLUMN_SIGNAL", 0)),
            "FS_PATTERN": _coerce_int01(l2.get("FS_PATTERN", 0)),
            "TB_PATTERN": _coerce_int01(l2.get("TB_PATTERN", 0)),
            "PARTIAL_FS_PATTERN": _coerce_int01(l2.get("PARTIAL_FS_PATTERN", 0)),
            "role_in_graph": l3.get("role_in_graph", "UNKNOWN") or "UNKNOWN",
            "outgoing_refs": list(l3.get("outgoing_refs", []) or []),
            "incoming_refs": list(l3.get("incoming_refs", []) or []),
            "consolidate": _coerce_bool(l3.get("consolidate", False)),
            "attention_boost": _coerce_bool(l3.get("attention_boost", False)),
            "gate_passed": _coerce_bool(l4.get("passed", True)),
            "blocked_by": l4.get("blocked_by"),
            "confidence": float(payload.get("layer5_confidence", 0.0) or 0.0),
            "main_sheet_confirmed": False,
            "is_main_source_hint": False,
            "title": _sheet_title_from_candidates(state, sheet_name),
        }

    canonical_sheet_evidence = parsed.get("sheet_evidence", {}) or {}
    if isinstance(canonical_sheet_evidence, dict):
        for sheet_name, payload in canonical_sheet_evidence.items():
            if isinstance(payload, dict):
                clean_name = str(sheet_name).strip()
                evidence_map[clean_name] = _make_sheet_record(clean_name, payload)

    legacy_evidence = parsed.get("evidence", {}) or {}
    if isinstance(legacy_evidence, dict):
        nested_sheet_evidence = legacy_evidence.get("sheet_evidence", {}) or {}
        if isinstance(nested_sheet_evidence, dict):
            for sheet_name, payload in nested_sheet_evidence.items():
                if isinstance(payload, dict):
                    clean_name = str(sheet_name).strip()
                    evidence_map[clean_name] = _make_sheet_record(clean_name, payload)

        for sheet_name, payload in legacy_evidence.items():
            if not isinstance(payload, dict):
                continue
            if sheet_name in {"ocr_used", "company_columns", "sections_found", "sheet_evidence"}:
                continue
            if (
                "layer1" in payload
                or "layer2" in payload
                or "layer3" in payload
                or "layer4" in payload
            ):
                clean_name = str(sheet_name).strip()
                evidence_map[clean_name] = _make_sheet_record(clean_name, payload)

    main_sheet = _clean_sheet_name(parsed.get("main_sheet_name"))
    nn = parsed.get("nn_evidence", {}) or {}
    if main_sheet and nn and main_sheet not in evidence_map:
        evidence_map[main_sheet] = _make_sheet_record(main_sheet, nn)

    main_source = _clean_sheet_name(parsed.get("main_source_sheet_name"))
    if main_source and main_source not in evidence_map:
        evidence_map[main_source] = {
            "sheet_name": main_source,
            "evidence_status": "referenced_only",
            "source_task_type": "research_agent",
            "COMPANY_COLUMN_SIGNAL": 0,
            "HIDDEN_SIGNAL": 0,
            "COA_SIGNAL": 0,
            "CONSOLIDATE_SIGNAL": 0,
            "CROSS_REF_SIGNAL": 0,
            "AJE_SIGNAL": 0,
            "FORMULA_SIGNAL": 0,
            "CODE_COLUMN_SIGNAL": 0,
            "FINAL_COLUMN_SIGNAL": 0,
            "FS_PATTERN": 0,
            "TB_PATTERN": 0,
            "PARTIAL_FS_PATTERN": 0,
            "role_in_graph": "UNKNOWN",
            "outgoing_refs": [],
            "incoming_refs": [],
            "consolidate": False,
            "attention_boost": False,
            "gate_passed": False,
            "blocked_by": "UNVALIDATED_REFERENCED_ONLY",
            "confidence": 0.0,
            "main_sheet_confirmed": False,
            "is_main_source_hint": True,
            "title": _sheet_title_from_candidates(state, main_source),
        }

    return evidence_map


def _build_sheet_evidence_index(state) -> dict[str, dict]:
    """
    Consolidate all evidence into a per-sheet map.

    For any discovered sheet not explicitly analyzed, create a conservative
    placeholder so missing evidence is never treated as a pass.
    """
    if state.get("sheet_evidence_index"):
        return state["sheet_evidence_index"]

    evidence_index: dict[str, dict] = {}

    # explicit evidence from task results
    for task in state.get("task_results", []):
        parsed = _parse_json_from_text(task.get("result", ""))
        if not parsed:
            continue
        task_map = _derive_sheet_evidence_from_single_task(parsed, state)
        for sheet, ev in task_map.items():
            evidence_index[sheet] = ev

    # placeholders for known sheets without explicit evidence
    for sheet in _extract_candidate_sheet_names_from_state(state):
        if sheet not in evidence_index:
            evidence_index[sheet] = {
                "sheet_name": sheet,
                "evidence_status": "missing",
                "source_task_type": "none",
                "COMPANY_COLUMN_SIGNAL": 0,
                "HIDDEN_SIGNAL": 0,
                "COA_SIGNAL": 0,
                "CONSOLIDATE_SIGNAL": 0,
                "CROSS_REF_SIGNAL": 0,
                "AJE_SIGNAL": 0,
                "FORMULA_SIGNAL": 0,
                "CODE_COLUMN_SIGNAL": 0,
                "FINAL_COLUMN_SIGNAL": 0,
                "FS_PATTERN": 0,
                "TB_PATTERN": 0,
                "PARTIAL_FS_PATTERN": 0,
                "role_in_graph": "UNKNOWN",
                "outgoing_refs": [],
                "incoming_refs": [],
                "consolidate": False,
                "attention_boost": False,
                "gate_passed": False,
                "blocked_by": "UNVALIDATED_NO_EVIDENCE",
                "confidence": 0.0,
                "main_sheet_confirmed": False,
                "is_main_source_hint": False,
                "title": _sheet_title_from_candidates(state, sheet),
            }

    state["sheet_evidence_index"] = evidence_index
    add_step_log(state, "evidence_index:built", {
        "sheet_count": len(evidence_index),
        "sheets": sorted(evidence_index.keys()),
    })
    return evidence_index


def _extract_nn_evidence(state, sheet_name: str | None) -> dict:
    if not sheet_name:
        return {}
    index = _build_sheet_evidence_index(state)
    return dict(index.get(sheet_name, {}))


# ─────────────────────────────────────────────────────────────────────────────
#  Validation and normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_research_agent_result(parsed: dict) -> tuple[bool, list[str]]:
    issues: list[str] = []

    name = _clean_sheet_name(parsed.get("main_sheet_name"))
    exists = _coerce_bool(parsed.get("main_sheet_exists", False))
    confidence = float(parsed.get("confidence", parsed.get("confidence_score", 0.0)) or 0.0)
    confirmed = _coerce_bool(parsed.get("main_sheet_confirmed", False))

    sheet_evidence = parsed.get("sheet_evidence", {}) or {}
    if exists and name and sheet_evidence and name not in sheet_evidence:
        issues.append("main_sheet_name missing from sheet_evidence")

    if exists and not name:
        issues.append("main_sheet_exists=true but main_sheet_name is empty")

    if name and confidence < 0.40 and exists:
        issues.append("confidence < 0.40 but main_sheet_exists=true")

    if confirmed and confidence < 0.70:
        issues.append("main_sheet_confirmed=true but confidence < 0.70")

    return (len(issues) == 0), issues

def _normalize_research_agent_result(parsed: dict) -> dict:
    fixed = dict(parsed)

    name = _clean_sheet_name(fixed.get("main_sheet_name"))
    fixed["main_sheet_name"] = name

    raw_conf = fixed.get("confidence", fixed.get("confidence_score", 0.0))
    confidence = float(raw_conf or 0.0)
    fixed["confidence"] = confidence

    legacy_evidence = fixed.get("evidence", {}) or {}
    if "sheet_evidence" not in fixed:
        if isinstance(legacy_evidence, dict) and isinstance(legacy_evidence.get("sheet_evidence"), dict):
            fixed["sheet_evidence"] = legacy_evidence["sheet_evidence"]
        else:
            fixed["sheet_evidence"] = {}

    confirmed = confidence >= 0.70
    exists = bool(name) and confidence >= 0.40

    nn = fixed.get("nn_evidence", {}) or {}
    l4 = nn.get("layer4", {}) or {}
    if l4 and (_coerce_bool(l4.get("passed", True)) is False):
        exists = False
        confirmed = False
        fixed["main_sheet_name"] = None
        fixed["confidence"] = 0.0

    fixed["main_sheet_exists"] = exists
    fixed["main_sheet_confirmed"] = confirmed

    if not exists:
        fixed["main_sheet_name"] = None
        fixed["main_sheet_confirmed"] = False
        fixed["confidence"] = 0.0

    return fixed

def _candidate_names_from_parsed_result(parsed: dict) -> set[str]:
    names: set[str] = set()
    if not parsed:
        return names

    for field in (
        "main_sheet_name",
        "technical_main_sheet",
        "business_main_sheet",
        "runner_up",
        "main_source_sheet_name",
        "strongest_candidate",
        "verification_target",
        "fallback_candidate",
    ):
        val = parsed.get(field)
        if val:
            names.add(str(val).strip())

    sheet_evidence = parsed.get("sheet_evidence", {}) or {}
    if isinstance(sheet_evidence, dict):
        for sheet in sheet_evidence.keys():
            if sheet:
                names.add(str(sheet).strip())

    legacy_evidence = parsed.get("evidence", {}) or {}
    if isinstance(legacy_evidence, dict):
        nested_sheet_evidence = legacy_evidence.get("sheet_evidence", {}) or {}
        if isinstance(nested_sheet_evidence, dict):
            for sheet in nested_sheet_evidence.keys():
                if sheet:
                    names.add(str(sheet).strip())

        for sheet, payload in legacy_evidence.items():
            if (
                sheet
                and isinstance(payload, dict)
                and sheet not in {"ocr_used", "company_columns", "sections_found", "sheet_evidence"}
            ):
                names.add(str(sheet).strip())

    return {n for n in names if n}


def _apply_sheet_payload(row: dict, payload: dict, state) -> None:
    l1 = payload.get("layer1", {}) or {}
    l2 = payload.get("layer2", {}) or {}
    l3 = payload.get("layer3", {}) or {}
    l4 = payload.get("layer4", {}) or {}

    row["evidence_status"] = "explicit"

    if "research_agent" not in row["source_task_types"]:
        row["source_task_types"].append("research_agent")

    row["COMPANY_COLUMN_SIGNAL"] = _coerce_int01(
        l1.get("COMPANY_COLUMN_SIGNAL", row["COMPANY_COLUMN_SIGNAL"])
    )
    row["HIDDEN_SIGNAL"] = _coerce_int01(
        l1.get("HIDDEN_SIGNAL", row["HIDDEN_SIGNAL"])
    )
    row["COA_SIGNAL"] = _coerce_int01(
        l1.get("COA_SIGNAL", row["COA_SIGNAL"])
    )
    row["CONSOLIDATE_SIGNAL"] = _coerce_int01(
        l1.get("CONSOLIDATE_SIGNAL", row["CONSOLIDATE_SIGNAL"])
    )
    row["CROSS_REF_SIGNAL"] = _coerce_int01(
        l1.get("CROSS_REF_SIGNAL", row["CROSS_REF_SIGNAL"])
    )
    row["AJE_SIGNAL"] = _coerce_int01(
        l1.get("AJE_SIGNAL", row["AJE_SIGNAL"])
    )
    row["FORMULA_SIGNAL"] = _coerce_int01(
        l1.get("FORMULA_SIGNAL", row["FORMULA_SIGNAL"])
    )
    row["CODE_COLUMN_SIGNAL"] = _coerce_int01(
        l1.get("CODE_COLUMN_SIGNAL", row["CODE_COLUMN_SIGNAL"])
    )
    row["FINAL_COLUMN_SIGNAL"] = _coerce_int01(
        l1.get("FINAL_COLUMN_SIGNAL", row["FINAL_COLUMN_SIGNAL"])
    )

    row["FS_PATTERN"] = _coerce_int01(
        l2.get("FS_PATTERN", row["FS_PATTERN"])
    )
    row["TB_PATTERN"] = _coerce_int01(
        l2.get("TB_PATTERN", row["TB_PATTERN"])
    )
    row["PARTIAL_FS_PATTERN"] = _coerce_int01(
        l2.get("PARTIAL_FS_PATTERN", row["PARTIAL_FS_PATTERN"])
    )

    row["role_in_graph"] = l3.get("role_in_graph", row["role_in_graph"]) or row["role_in_graph"]
    row["outgoing_refs"] = list(l3.get("outgoing_refs", row["outgoing_refs"]) or [])
    row["incoming_refs"] = list(l3.get("incoming_refs", row["incoming_refs"]) or [])
    row["consolidate"] = _coerce_bool(
        l3.get("consolidate", row["consolidate"])
    )
    row["attention_boost"] = _coerce_bool(
        l3.get("attention_boost", row["attention_boost"])
    )

    row["gate_passed"] = _coerce_bool(
        l4.get("passed", row["gate_passed"])
    )
    row["blocked_by"] = l4.get("blocked_by", row["blocked_by"])

    row["confidence"] = float(
        payload.get("layer5_confidence", row["confidence"]) or 0.0
    )

    if not row.get("title"):
        row["title"] = _sheet_title_from_candidates(state, row["sheet_name"])


def _build_candidate_registry(state) -> dict[str, dict]:
    if state.get("candidate_registry"):
        return state["candidate_registry"]

    registry: dict[str, dict] = {}

    def ensure_row(sheet_name: str) -> dict:
        clean = str(sheet_name).strip()
        if not clean:
            raise ValueError("sheet_name must be non-empty")

        if clean not in registry:
            registry[clean] = {
                "sheet_name": clean,

                # provenance
                "seen_in_detector": False,
                "seen_in_output_candidates": False,
                "seen_in_source_candidates": False,
                "seen_in_profiles": False,
                "seen_as_main_sheet": False,
                "seen_as_technical_main_sheet": False,
                "seen_as_business_main_sheet": False,
                "seen_as_runner_up": False,
                "seen_as_main_source_sheet": False,

                # evidence state
                "evidence_status": "missing",  # explicit | referenced_only | missing
                "source_task_types": [],

                # L1
                "COMPANY_COLUMN_SIGNAL": 0,
                "HIDDEN_SIGNAL": 0,
                "COA_SIGNAL": 0,
                "CONSOLIDATE_SIGNAL": 0,
                "CROSS_REF_SIGNAL": 0,
                "AJE_SIGNAL": 0,
                "FORMULA_SIGNAL": 0,
                "CODE_COLUMN_SIGNAL": 0,
                "FINAL_COLUMN_SIGNAL": 0,

                # L2
                "FS_PATTERN": 0,
                "TB_PATTERN": 0,
                "PARTIAL_FS_PATTERN": 0,

                # L3
                "role_in_graph": "UNKNOWN",
                "outgoing_refs": [],
                "incoming_refs": [],
                "consolidate": False,
                "attention_boost": False,

                # L4/L5
                "gate_passed": False,
                "blocked_by": "UNVALIDATED_NO_EVIDENCE",
                "confidence": 0.0,

                # misc
                "main_sheet_confirmed": False,
                "is_main_source_hint": False,
                "title": _sheet_title_from_candidates(state, clean),

                # derived later
                "sheet_type": "UNKNOWN",
                "disqualification_class": "CRITICAL",
                "candidate_score": None,
            }
        return registry[clean]

    # 1) Detector/main_sheet_result candidates
    msr = state.get("main_sheet_result", {}) or {}

    detector_candidate = _clean_sheet_name(state.get("detector_candidate"))
    if detector_candidate:
        ensure_row(detector_candidate)["seen_in_detector"] = True

    for item in msr.get("output_candidates", []) or []:
        sheet = _clean_sheet_name(item.get("sheet") or item.get("sheet_name"))
        if not sheet:
            continue
        row = ensure_row(sheet)
        row["seen_in_output_candidates"] = True
        if not row.get("title"):
            row["title"] = str(item.get("title") or "")

    for item in msr.get("source_candidates", []) or []:
        sheet = _clean_sheet_name(item.get("sheet") or item.get("sheet_name"))
        if not sheet:
            continue
        row = ensure_row(sheet)
        row["seen_in_source_candidates"] = True
        if not row.get("title"):
            row["title"] = str(item.get("title") or "")

    for item in msr.get("profiles", []) or []:
        sheet = _clean_sheet_name(item.get("sheet") or item.get("sheet_name"))
        if not sheet:
            continue
        row = ensure_row(sheet)
        row["seen_in_profiles"] = True
        if not row.get("title"):
            row["title"] = str(item.get("title") or "")

    # 2) Parse all task results and register all names first
    parsed_tasks: list[dict] = []
    for task in state.get("task_results", []):
        parsed = _parse_json_from_text(task.get("result", ""))
        if not parsed:
            continue
        parsed_tasks.append(parsed)

        for sheet in _candidate_names_from_parsed_result(parsed):
            ensure_row(sheet)

        v = _clean_sheet_name(parsed.get("main_sheet_name"))
        if v:
            ensure_row(v)["seen_as_main_sheet"] = True

        v = _clean_sheet_name(parsed.get("technical_main_sheet"))
        if v:
            ensure_row(v)["seen_as_technical_main_sheet"] = True

        v = _clean_sheet_name(parsed.get("business_main_sheet"))
        if v:
            ensure_row(v)["seen_as_business_main_sheet"] = True

        v = _clean_sheet_name(parsed.get("runner_up"))
        if v:
            ensure_row(v)["seen_as_runner_up"] = True

        v = _clean_sheet_name(parsed.get("main_source_sheet_name"))
        if v:
            row = ensure_row(v)
            row["seen_as_main_source_sheet"] = True
            row["is_main_source_hint"] = True
            if row["evidence_status"] == "missing":
                row["evidence_status"] = "referenced_only"
                row["blocked_by"] = "UNVALIDATED_REFERENCED_ONLY"

    # 3) Merge explicit evidence from all known schema variants
    for parsed in parsed_tasks:
        canonical_sheet_evidence = parsed.get("sheet_evidence", {}) or {}
        if isinstance(canonical_sheet_evidence, dict):
            for sheet_name, payload in canonical_sheet_evidence.items():
                if isinstance(payload, dict):
                    _apply_sheet_payload(ensure_row(sheet_name), payload, state)

        legacy_evidence = parsed.get("evidence", {}) or {}
        if isinstance(legacy_evidence, dict):
            nested_sheet_evidence = legacy_evidence.get("sheet_evidence", {}) or {}
            if isinstance(nested_sheet_evidence, dict):
                for sheet_name, payload in nested_sheet_evidence.items():
                    if isinstance(payload, dict):
                        _apply_sheet_payload(ensure_row(sheet_name), payload, state)

            for sheet_name, payload in legacy_evidence.items():
                if not isinstance(payload, dict):
                    continue
                if sheet_name in {"ocr_used", "company_columns", "sections_found", "sheet_evidence"}:
                    continue
                if (
                    "layer1" in payload
                    or "layer2" in payload
                    or "layer3" in payload
                    or "layer4" in payload
                ):
                    _apply_sheet_payload(ensure_row(sheet_name), payload, state)

        # older single-winner nn_evidence fallback
        main_sheet = _clean_sheet_name(parsed.get("main_sheet_name"))
        nn = parsed.get("nn_evidence", {}) or {}
        if main_sheet and nn and (
            "layer1" in nn or "layer2" in nn or "layer3" in nn or "layer4" in nn
        ):
            _apply_sheet_payload(ensure_row(main_sheet), nn, state)

    # 4) Derive semantic fields
    for sheet, row in registry.items():
        row["sheet_type"] = _classify_sheet_type(
            sheet,
            row,
            row.get("title", ""),
        )
        row["disqualification_class"] = _disqualification_class(
            row,
            row["sheet_type"],
        )

    state["candidate_registry"] = registry

    add_step_log(state, "candidate_registry:built", {
        "sheet_count": len(registry),
        "sheets": sorted(registry.keys()),
        "summary": {
            name: {
                "evidence_status": row["evidence_status"],
                "sheet_type": row["sheet_type"],
                "role_in_graph": row["role_in_graph"],
                "gate_passed": row["gate_passed"],
                "blocked_by": row["blocked_by"],
                "confidence": row["confidence"],
                "seen_as_main_sheet": row["seen_as_main_sheet"],
                "seen_as_main_source_sheet": row["seen_as_main_source_sheet"],
            }
            for name, row in registry.items()
        },
    })

    return registry


def _score_candidate_row(row: dict) -> tuple:
    explicit = 1 if row.get("evidence_status") == "explicit" else 0
    reporting = 1 if row.get("sheet_type") == "REPORTING_FS" else 0
    passed = 1 if row.get("gate_passed", False) else 0
    noncritical = 1 if row.get("disqualification_class") in ("NONE", "TECHNICAL") else 0
    not_tb = 1 if (
        row.get("sheet_type") != "SOURCE_TB"
        and row.get("TB_PATTERN", 0) == 0
        and row.get("role_in_graph") != "TB"
    ) else 0

    fs = int(row.get("FS_PATTERN", 0) == 1)
    partial = int(row.get("PARTIAL_FS_PATTERN", 0) == 1)
    company = int(row.get("COMPANY_COLUMN_SIGNAL", 0) == 1)
    cross = int(row.get("CROSS_REF_SIGNAL", 0) == 1)
    detector = int(row.get("seen_in_detector", False))
    main_sheet_hint = int(row.get("seen_as_main_sheet", False))
    conf = float(row.get("confidence", 0.0) or 0.0)

    return (
        explicit,
        reporting,
        passed,
        noncritical,
        not_tb,
        fs,
        partial,
        company,
        cross,
        main_sheet_hint,
        detector,
        conf,
    )

def _restrict_synthesis_to_evidence_backed_candidates(parsed: dict, state) -> dict:
    """
    Synthesis may only select sheets with explicit evidence.
    Referenced-only or missing-evidence sheets cannot become main sheet.
    """
    fixed = dict(parsed)
    candidate = _clean_sheet_name(fixed.get("main_sheet_name"))
    if not candidate:
        return fixed

    ev = _extract_nn_evidence(state, candidate)
    if ev.get("evidence_status") == "explicit":
        return fixed

    add_step_log(state, "synthesis:invalid_candidate_rejected", {
        "candidate": candidate,
        "evidence_status": ev.get("evidence_status"),
        "blocked_by": ev.get("blocked_by"),
    })

    fixed["main_sheet_exists"] = False
    fixed["main_sheet_name"] = None
    fixed["confidence"] = 0.0
    fixed["reasoning"] = (
        f"Synthesis proposed '{candidate}', but that sheet has no explicit NN evidence. "
        f"Only explicitly analyzed sheets may become main_sheet_name."
    )
    fixed["api_response"] = {
        "main_sheet_exists": False,
        "main_sheet_name": None,
    }
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
#  NN-aware guardrails
# ─────────────────────────────────────────────────────────────────────────────

def _apply_nn_guardrails(parsed: dict, state) -> dict:
    result = dict(parsed)
    sheet_name = _clean_sheet_name(result.get("main_sheet_name"))
    confidence = float(result.get("confidence", 0.0) or 0.0)

    if not result.get("main_sheet_exists") or not sheet_name:
        result["main_sheet_exists"] = False
        result["main_sheet_name"] = None
        result["confidence"] = 0.0
        result["api_response"] = {
            "main_sheet_exists": False,
            "main_sheet_name": None,
        }
        return result

    ev = _extract_nn_evidence(state, sheet_name)
    disqualify_reason: str | None = None
    is_critical_block = False

    if not ev or ev.get("evidence_status") != "explicit":
        disqualify_reason = (
            f"G0: no explicit NN evidence exists for sheet '{sheet_name}' "
            f"(evidence_status={ev.get('evidence_status', 'missing') if ev else 'missing'})"
        )
        is_critical_block = True

    elif not ev.get("gate_passed", False):
        blocked_by = ev.get("blocked_by", "unknown gate")

        if blocked_by == "GATE_2":
            add_step_log(state, "guardrail:gate2_noted", {
                "sheet": sheet_name,
                "reason": (
                    "GATE_2 (no company columns) — recorded for Layer-6 business "
                    "arbitration, not treated as fatal at this stage"
                ),
            })
        else:
            disqualify_reason = (
                f"G4: NN layer4.passed=false (blocked_by={blocked_by}) "
                f"— critical gate fired, sheet permanently disqualified"
            )
            is_critical_block = True

    elif ev.get("HIDDEN_SIGNAL") == 1:
        disqualify_reason = "G1: HIDDEN_SIGNAL=1 — hidden sheets cannot be main sheet"
        is_critical_block = True

    elif ev.get("TB_PATTERN") == 1:
        disqualify_reason = "G3: TB_PATTERN=1 — trial balance / ledger sheets are sources, not main sheets"
        is_critical_block = True

    elif confidence < 0.40:
        disqualify_reason = f"G5: confidence={confidence:.2f} is below 0.40 threshold"

    if not disqualify_reason:
        return result

    add_step_log(state, "guardrail:disqualify", {
        "sheet": sheet_name,
        "reason": disqualify_reason,
        "critical": is_critical_block,
        "nn_evidence": safe_preview(ev, 800),
    })

    runner_up = _clean_sheet_name(result.get("runner_up"))
    if runner_up:
        ru_ev = _extract_nn_evidence(state, runner_up)
        ru_conf = float(ru_ev.get("confidence", 0.0) or 0.0)

        ru_passes = (
            ru_ev.get("evidence_status") == "explicit"
            and ru_ev.get("gate_passed", False)
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
                "main_sheet_name": runner_up,
                "confidence": ru_conf,
                "reasoning": (
                    f"Sheet '{sheet_name}' was blocked by Python guardrails "
                    f"({disqualify_reason}). Runner-up '{runner_up}' had explicit "
                    f"NN evidence and passed all guardrails."
                ),
                "api_response": {
                    "main_sheet_exists": True,
                    "main_sheet_name": runner_up,
                },
            })
            return result

    add_step_log(state, "guardrail:no_valid_sheet", {
        "reason": disqualify_reason,
        "runner_up_attempted": runner_up,
        "runner_up_passed": False,
    })
    result.update({
        "main_sheet_exists": False,
        "main_sheet_name": None,
        "confidence": 0.0,
        "reasoning": (
            f"No valid main sheet found. '{sheet_name}' was blocked: "
            f"{disqualify_reason}. Runner-up '{runner_up}' was absent, invalid, "
            f"or lacked explicit NN evidence."
        ),
        "api_response": {
            "main_sheet_exists": False,
            "main_sheet_name": None,
        },
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 6 — Business arbitration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_sheet_type(sheet_name: str | None, ev: dict, title: str = "") -> str:
    if ev.get("evidence_status") != "explicit":
        return "UNKNOWN"

    sig = _business_signals(sheet_name, ev, title)
    name_l = _norm_text(sheet_name)

    if ev.get("TB_PATTERN") == 1 or ev.get("role_in_graph") == "TB":
        return "SOURCE_TB"

    if ev.get("FS_PATTERN") == 1:
        if sig["STAGING_SHEET_SIGNAL"] and any(
            kw in name_l for kw in ["aje", "adjusting", "adjustments", "elimination"]
        ):
            return "ADJUSTMENT_STAGING"
        return "REPORTING_FS"

    if (
        ev.get("AJE_SIGNAL") == 1
        or sig["STAGING_SHEET_SIGNAL"]
    ):
        return "ADJUSTMENT_STAGING"

    if sig["FINAL_OUTPUT_ROLE_SIGNAL"]:
        return "REPORTING_FS"

    if ev.get("consolidate", False) or ev.get("role_in_graph") == "INTERMEDIATE":
        return "INTERMEDIATE_CONSOLIDATION"

    if ev.get("COA_SIGNAL") == 1 and ev.get("CROSS_REF_SIGNAL") == 0:
        return "AUXILIARY_SCHEDULE"

    return "UNKNOWN"

def _disqualification_class(ev: dict, sheet_type: str) -> str:
    evidence_status = ev.get("evidence_status")
    blocked_by = ev.get("blocked_by")
    passed = ev.get("gate_passed", False)

    if evidence_status != "explicit":
        return "CRITICAL"

    if passed and not blocked_by:
        return "NONE"

    if blocked_by in ("GATE_1", "GATE_3", "GATE_4"):
        return "CRITICAL"

    if blocked_by == "GATE_2" and sheet_type == "REPORTING_FS":
        return "TECHNICAL"

    return "CRITICAL"


def _candidate_sheet_names(state) -> list[str]:
    return list(_build_sheet_evidence_index(state).keys())


def _pick_promat_fallback_candidate(state) -> tuple[str | None, dict]:
    candidates = _candidate_sheet_names(state)
    if not candidates:
        return None, {}

    ranked = []
    for sheet in candidates:
        ev = _extract_nn_evidence(state, sheet)
        title = _sheet_title_from_candidates(state, sheet)
        ranked.append((sheet, _presentation_rank(sheet, ev, title), ev))

    ranked.sort(key=lambda x: x[1], reverse=True)

    best_sheet, _, best_ev = ranked[0]
    return best_sheet, best_ev

def _presentation_rank(sheet: str, ev: dict, title: str) -> tuple:
    sheet_type = _classify_sheet_type(sheet, ev, title)
    dq = _disqualification_class(ev, sheet_type)
    sig = _business_signals(sheet, ev, title)

    explicit = 1 if ev.get("evidence_status") == "explicit" else 0
    non_critical = 1 if dq in ("NONE", "TECHNICAL") else 0
    final_output = 1 if sig["FINAL_OUTPUT_ROLE_SIGNAL"] else 0
    canonical_title = 1 if sig["CANONICAL_FS_TITLE_SIGNAL"] else 0
    presentation_layout = 1 if sig["PRESENTATION_LAYOUT_SIGNAL"] else 0
    passed_gate = 1 if ev.get("gate_passed", False) else 0
    reporting_fs = 1 if sheet_type == "REPORTING_FS" else 0
    not_staging = 1 if sheet_type != "ADJUSTMENT_STAGING" else 0
    conf = float(ev.get("confidence", 0.0) or 0.0)

    return (
        explicit,
        non_critical,
        final_output,
        canonical_title,
        presentation_layout,
        passed_gate,
        reporting_fs,
        not_staging,
        conf,
    )


def _pick_presentation_candidate(state) -> tuple[str | None, dict]:
    best_name = None
    best_meta = {}
    best_rank = None

    for sheet in _candidate_sheet_names(state):
        ev = _extract_nn_evidence(state, sheet)
        title = _sheet_title_from_candidates(state, sheet)
        sheet_type = _classify_sheet_type(sheet, ev, title)
        dq = _disqualification_class(ev, sheet_type)
        sig = _business_signals(sheet, ev, title)

        rank = _presentation_rank(sheet, ev, title)

        if rank[0] == 0:
            continue
        if dq == "CRITICAL":
            continue
        if sheet_type != "REPORTING_FS":
            continue

        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_name = sheet
            best_meta = {
                "sheet_type": sheet_type,
                "blocked_by": ev.get("blocked_by"),
                "disqualification_class": dq,
                "confidence": float(ev.get("confidence", 0.0) or 0.0),
                "title": title,
                "evidence_status": ev.get("evidence_status"),
                "signals": sig,
            }

    return best_name, best_meta

def _apply_business_arbitration(parsed: dict, state) -> dict:
    result = dict(parsed)

    technical_winner = _clean_sheet_name(result.get("main_sheet_name"))
    if not technical_winner:
        result.setdefault("technical_main_sheet", None)
        result.setdefault("presentation_main_sheet", None)
        result.setdefault("business_main_sheet", None)
        result["decision_mode"] = "no_valid_sheet"
        result["business_arbitration"] = {
            "technical_winner_sheet_type": None,
            "presentation_candidate": None,
            "presentation_candidate_sheet_type": None,
            "presentation_candidate_blocked_by": None,
            "presentation_candidate_disqualification_class": None,
            "override_applied": False,
        }
        return result

    tech_ev = _extract_nn_evidence(state, technical_winner)
    tech_title = _sheet_title_from_candidates(state, technical_winner)
    tech_type = _classify_sheet_type(technical_winner, tech_ev, tech_title)
    tech_conf = float(tech_ev.get("confidence", 0.0) or 0.0)

    presentation_candidate, pc_meta = _pick_presentation_candidate(state)
    pc_type = pc_meta.get("sheet_type")
    pc_blocked = pc_meta.get("blocked_by")
    pc_dq = pc_meta.get("disqualification_class")
    pc_conf = float(pc_meta.get("confidence", 0.0) or 0.0)
    pc_signals = pc_meta.get("signals", {}) or {}

    override = False

    if presentation_candidate and presentation_candidate != technical_winner:
        safe_candidate = (
            pc_dq in ("NONE", "TECHNICAL")
            and pc_blocked not in ("GATE_1", "GATE_3", "GATE_4")
            and pc_type == "REPORTING_FS"
            and pc_signals.get("FINAL_OUTPUT_ROLE_SIGNAL") is True
        )

        technical_is_less_human_final = tech_type in (
            "ADJUSTMENT_STAGING",
            "INTERMEDIATE_CONSOLIDATION",
        )

        presentation_materially_better = (
            pc_signals.get("CANONICAL_FS_TITLE_SIGNAL") is True
            and pc_signals.get("PRESENTATION_LAYOUT_SIGNAL") is True
            and tech_conf <= 0.55
        )

        override = safe_candidate and (
            technical_is_less_human_final or presentation_materially_better
        )

    authoritative_sheet = technical_winner
    presentation_sheet = presentation_candidate if presentation_candidate else technical_winner
    final_sheet = presentation_sheet if override else authoritative_sheet

    result["technical_main_sheet"] = authoritative_sheet
    result["presentation_main_sheet"] = presentation_sheet
    result["business_main_sheet"] = presentation_sheet
    result["main_sheet_name"] = final_sheet
    result["main_sheet_exists"] = bool(final_sheet)
    result["decision_mode"] = (
        "business_override" if override else "technical_default"
    )
    result["api_response"] = {
        "main_sheet_exists": bool(final_sheet),
        "main_sheet_name": final_sheet,
    }

    result["business_arbitration"] = {
        "technical_winner_sheet_type": tech_type,
        "presentation_candidate": presentation_candidate,
        "presentation_candidate_sheet_type": pc_type,
        "presentation_candidate_blocked_by": pc_blocked,
        "presentation_candidate_disqualification_class": pc_dq,
        "override_applied": override,
    }

    if override:
        result["reasoning"] = (
            f"Technical winner '{technical_winner}' remains the authoritative "
            f"structural FS sheet, but Layer-6 selected '{presentation_candidate}' "
            f"as the human-facing final report sheet."
        )
    else:
        result["reasoning"] = (
            f"Technical winner '{technical_winner}' remained final. "
            f"Presentation candidate was '{presentation_candidate}', "
            f"but override conditions were not met."
        )

    add_step_log(state, "business_arbitration:resolved", {
        "technical_winner": technical_winner,
        "technical_type": tech_type,
        "presentation_candidate": presentation_candidate,
        "presentation_type": pc_type,
        "presentation_disqualification": pc_dq,
        "override": override,
        "final_sheet": final_sheet,
    })

    return result

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


def _parse_l6_judge_verdict(j_out: str) -> str:
    for pat in [r'"verdict"\s*:\s*"([^"]+)"', r"'verdict'\s*:\s*'([^']+)'"]:
        m = re.search(pat, j_out)
        if m:
            return m.group(1)
    return "approve_transfer"


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
        agent_name=agent_name,
        agent_output=agent_output,
        plaintiff_charges=po,
        defense_arguments=do,
    )
    jo = _invoke_agent(j, jp)
    add_agent_log(state, f"court_j_{attempt}", "judge_agent", jp, jo)

    verdict = _parse_judge_verdict(jo)
    add_court_log(state, agent_name, attempt, po, do, jo, verdict)
    return verdict, jo


def run_l6_court_session(state, l5_payload: dict, l6_payload: dict):
    try:
        p = build_l6_plaintiff_agent()
        d = build_l6_defense_agent()
        j = build_l6_judge_agent()

        l5_text = json.dumps(l5_payload, ensure_ascii=False)
        l6_text = json.dumps(l6_payload, ensure_ascii=False)

        pp = build_l6_court_user_prompt(l5_text, l6_text)
        po = _invoke_agent(p, pp)
        add_agent_log(state, "l6_court_p", "l6_plaintiff_agent", pp, po)

        dp = build_l6_court_user_prompt(l5_text, l6_text, plaintiff_charges=po)
        do = _invoke_agent(d, dp)
        add_agent_log(state, "l6_court_d", "l6_defense_agent", dp, do)

        jp = build_l6_court_user_prompt(
            l5_text,
            l6_text,
            plaintiff_charges=po,
            defense_arguments=do,
        )
        jo = _invoke_agent(j, jp)
        add_agent_log(state, "l6_court_j", "l6_judge_agent", jp, jo)

        verdict = _parse_l6_judge_verdict(jo)
        add_court_log(state, "layer6_transfer", 1, po, do, jo, verdict)

        return verdict, jo

    except Exception as e:
        add_error_log(state, "run_l6_court_session", e)
        add_step_log(state, "layer6_court:fallback", {
            "reason": str(e),
            "fallback_verdict": "court_unavailable",
        })
        return "court_unavailable", None


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
        llm = get_llm()
        file_path = state["user_input"]
        add_step_log(state, "analyze:start", {"file_path": file_path})

        excel_summary = inspect_workbook(file_path)
        main_sheet_result = detect_main_sheet(file_path)
        detector_cand = main_sheet_result.get("main_sheet")

        prompt = build_analyze_prompt(
            excel_summary=excel_summary,
            main_sheet_result=main_sheet_result,
        )
        result = llm.invoke(prompt)

        add_step_log(state, "analyze:end", {"detector_candidate": detector_cand})

        return {
            **state,
            "excel_summary": excel_summary,
            "main_sheet_result": main_sheet_result,
            "detector_candidate": detector_cand,
            "main_sheet_name": None,
            "has_main_sheet": False,
            "analysis": result.content,
            "next_step": "plan",
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
        tasks = [{
            "id": "task_1",
            "agent": "research_agent",
            "instruction": task_instruction,
        }]
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
                parsed = _parse_json_from_text(out)

                if parsed:
                    parsed = _normalize_research_agent_result(parsed)
                    ok, issues = _validate_research_agent_result(parsed)

                    add_step_log(state, "act:research_validation", {
                        "task_id": tid,
                        "valid": ok,
                        "issues": issues,
                        "normalized_main_sheet": parsed.get("main_sheet_name"),
                        "normalized_exists": parsed.get("main_sheet_exists"),
                        "normalized_confidence": parsed.get("confidence"),
                    })

                    out = json.dumps(parsed, ensure_ascii=False)
                else:
                    add_step_log(state, "act:research_validation", {
                        "task_id": tid,
                        "valid": False,
                        "issues": ["research agent output could not be parsed as JSON"],
                    })

                add_agent_log(state, tid, aname, instr, response=out)
                results.append({"task_id": tid, "agent": aname, "result": out})

            except Exception as ae:
                add_agent_log(state, tid, aname, instr, error=str(ae))
                add_error_log(state, f"act:{aname}", ae)
                results.append({
                    "task_id": tid,
                    "agent": aname,
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
            meta: list[dict] = []
            attempt = 1
            final_v = "approved"
            final_j = ""

            while attempt <= MAX_COURT_RETRIES + 1:
                v, jo = run_court_session(state, aname, cur, attempt)
                final_v, final_j = v, jo
                meta.append({
                    "attempt": attempt,
                    "verdict": v,
                    "judge_summary": safe_preview(jo, 400),
                })

                if v in ("approved", "approved_with_note"):
                    break

                if attempt > MAX_COURT_RETRIES:
                    add_step_log(state, "court:max_retries", {
                        "task_id": tid,
                        "verdict": v,
                    })
                    break

                if v in ("revise_and_retry", "reject"):
                    try:
                        rp = build_agent_revision_prompt(aname, cur, jo)
                        rr = build_research_agent().invoke(
                            {"messages": [{"role": "user", "content": rp}]}
                        )
                        cur = rr["messages"][-1].content

                        parsed = _parse_json_from_text(cur)
                        if parsed:
                            parsed = _normalize_research_agent_result(parsed)
                            cur = json.dumps(parsed, ensure_ascii=False)

                        add_agent_log(state, f"revision_{attempt}", aname, rp, cur)
                    except Exception as re_err:
                        add_error_log(state, f"court:revision_{attempt}", re_err)
                        break

                attempt += 1

            reviewed.append({
                "task_id": tid,
                "agent": aname,
                "result": cur,
                "court_verdict": final_v,
                "court_judge_output": final_j,
                "court_sessions": meta,
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

    Hard requirement:
    Always return a main sheet answer if any candidate sheet exists in the workbook,
    even when the system is uncertain.

    Resolution order:
    1. Build evidence index.
    2. Call synthesis LLM.
    3. Parse output.
    4. Restrict synthesis to evidence-backed candidates when possible.
    5. Apply deterministic Python guardrails.
    6. Apply Layer-6 business arbitration.
    7. Run Layer-6 court in best-effort mode.
    8. If no valid final sheet survives, force-pick the best PROMAT-consistent fallback.
    """
    try:
        llm = get_llm()

        evidence_index = _build_sheet_evidence_index(state)
        explicit_candidates = sorted(
            s for s, ev in evidence_index.items()
            if ev.get("evidence_status") == "explicit"
        )
        all_candidates = sorted(evidence_index.keys())

        add_step_log(state, "synthesize:start", {
            "detector_candidate": state.get("detector_candidate"),
            "task_count": len(state.get("task_results", [])),
            "explicit_candidates": explicit_candidates,
            "all_candidates": all_candidates,
        })

        def _build_forced_fallback_result(reason: str) -> dict:
            chosen_name, chosen_ev = _pick_promat_fallback_candidate(state)
            alt_candidates = []
            for sheet in _candidate_sheet_names(state):
                ev = _extract_nn_evidence(state, sheet)
                title = _sheet_title_from_candidates(state, sheet)
                stype = _classify_sheet_type(sheet, ev, title)
                if (
                    ev.get("evidence_status") == "explicit"
                    and stype == "REPORTING_FS"
                    and ev.get("HIDDEN_SIGNAL", 0) == 0
                    and ev.get("TB_PATTERN", 0) == 0
                ):
                    alt_candidates.append((sheet, ev))

            chosen_type = _classify_sheet_type(
                chosen_name,
                chosen_ev,
                _sheet_title_from_candidates(state, chosen_name),
            )

            if chosen_type == "SOURCE_TB" and alt_candidates:
                alt_candidates.sort(key=lambda x: float(x[1].get("confidence", 0.0) or 0.0), reverse=True)
                chosen_name, chosen_ev = alt_candidates[0]
                
            if not chosen_name:
                return {
                    "main_sheet_exists": False,
                    "main_sheet_name": None,
                    "technical_main_sheet": None,
                    "business_main_sheet": None,
                    "decision_mode": "no_valid_sheet",
                    "confidence": 0.0,
                    "reasoning": f"No sheet candidates were available at all. {reason}",
                    "api_response": {
                        "main_sheet_exists": False,
                        "main_sheet_name": None,
                    },
                    "business_arbitration": {
                        "technical_winner_sheet_type": None,
                        "business_candidate": None,
                        "business_candidate_sheet_type": None,
                        "business_candidate_blocked_by": None,
                        "business_candidate_disqualification_class": None,
                        "override_applied": False,
                    },
                    "layer6_court_status": "not_run",
                    "fallback_used": True,
                    "fallback_reason": reason,
                }

            stype = _classify_sheet_type(
                chosen_name,
                chosen_ev,
                _sheet_title_from_candidates(state, chosen_name),
            )
            dq = _disqualification_class(chosen_ev, stype)

            fallback_conf = float(chosen_ev.get("confidence", 0.0) or 0.01)
            if fallback_conf <= 0.0:
                fallback_conf = 0.01

            add_step_log(state, "synthesize:forced_fallback", {
                "chosen_name": chosen_name,
                "reason": reason,
                "sheet_type": stype,
                "disqualification_class": dq,
                "evidence_status": chosen_ev.get("evidence_status"),
                "confidence": fallback_conf,
            })

            return {
                "main_sheet_exists": True,
                "main_sheet_name": chosen_name,
                "technical_main_sheet": chosen_name,
                "business_main_sheet": chosen_name,
                "decision_mode": "forced_fallback",
                "confidence": fallback_conf,
                "reasoning": (
                    f"No fully validated final sheet survived synthesis/guardrails, "
                    f"so the system returned the best PROMAT-consistent fallback "
                    f"candidate '{chosen_name}'. Reason: {reason}"
                ),
                "api_response": {
                    "main_sheet_exists": True,
                    "main_sheet_name": chosen_name,
                },
                "business_arbitration": {
                    "technical_winner_sheet_type": stype,
                    "business_candidate": chosen_name,
                    "business_candidate_sheet_type": stype,
                    "business_candidate_blocked_by": chosen_ev.get("blocked_by"),
                    "business_candidate_disqualification_class": dq,
                    "override_applied": False,
                },
                "layer6_court_status": "not_run",
                "fallback_used": True,
                "fallback_reason": reason,
            }

        raw_content = ""
        name = None
        exists = False
        confidence = 0.0
        parsed = {}

        try:
            prompt = build_synthesize_prompt(
                analysis=state.get("analysis", ""),
                plan=state.get("plan", ""),
                main_sheet_result=state.get("main_sheet_result", {}),
                task_results=state.get("task_results", []),
            )
            llm_result = llm.invoke(prompt)
            raw_content = llm_result.content

            name, exists, confidence, parsed = _parse_synthesis_result(raw_content)
            add_step_log(state, "synthesize:parsed", {
                "raw_name": name,
                "raw_exists": exists,
                "confidence": confidence,
                "detector_was": state.get("detector_candidate"),
            })

        except Exception as synth_err:
            add_error_log(state, "synthesize:llm_invoke", synth_err)
            parsed = {}
            add_step_log(state, "synthesize:llm_fallback", {
                "reason": str(synth_err),
            })

        final_result: dict

        if parsed:
            parsed = _restrict_synthesis_to_evidence_backed_candidates(parsed, state)
            guarded = _apply_nn_guardrails(parsed, state)
            l5_payload = dict(guarded)
            arbitrated = _apply_business_arbitration(guarded, state)

            try:
                l6_verdict, l6_judge_output = run_l6_court_session(
                    state,
                    l5_payload=l5_payload,
                    l6_payload=arbitrated,
                )
            except Exception as l6_err:
                add_error_log(state, "synthesize:run_l6_court_session", l6_err)
                add_step_log(state, "layer6_court:fallback", {
                    "reason": str(l6_err),
                    "fallback_verdict": "court_unavailable",
                    "technical_main_sheet": l5_payload.get("main_sheet_name"),
                    "final_after_python_arbitration": arbitrated.get("main_sheet_name"),
                })
                l6_verdict = "court_unavailable"
                l6_judge_output = None

            add_step_log(state, "layer6_court:verdict", {
                "verdict": l6_verdict,
                "technical_main_sheet": l5_payload.get("main_sheet_name"),
                "final_after_l6": arbitrated.get("main_sheet_name"),
                "judge_output_preview": safe_preview(l6_judge_output, 400),
            })

            if l6_verdict == "reject_transfer":
                technical_name = l5_payload.get("main_sheet_name")
                fallback = dict(l5_payload)
                fallback["technical_main_sheet"] = technical_name
                fallback["business_main_sheet"] = technical_name
                fallback["decision_mode"] = "technical_default_after_l6_court"
                fallback["layer6_court_status"] = "rejected_transfer"
                fallback["business_arbitration"] = {
                    "technical_winner_sheet_type": (
                        _apply_business_arbitration(guarded, state)
                        .get("business_arbitration", {})
                        .get("technical_winner_sheet_type")
                    ),
                    "business_candidate": None,
                    "business_candidate_sheet_type": None,
                    "business_candidate_blocked_by": None,
                    "business_candidate_disqualification_class": None,
                    "override_applied": False,
                }
                fallback["reasoning"] = (
                    f"Layer-6 court rejected the L5 → L6 transfer. "
                    f"The system reverted to the Layer-5 technical result "
                    f"'{technical_name}'."
                )
                final_result = fallback

            elif l6_verdict == "court_unavailable":
                final_result = dict(arbitrated)
                final_result["layer6_court_status"] = "unavailable"
                final_result["layer6_court_judge_output"] = None
                final_result["reasoning"] = (
                    f"{final_result.get('reasoning', '')} "
                    f"Layer-6 court was unavailable due to a model/server error, "
                    f"so the system kept the deterministic Python arbitration result."
                ).strip()

            else:
                final_result = dict(arbitrated)
                final_result["layer6_court_status"] = "approved"
                final_result["layer6_court_judge_output"] = l6_judge_output

        else:
            final_result = _build_forced_fallback_result(
                "Synthesis LLM did not return a parseable JSON result."
            )

        final_name = _clean_sheet_name(final_result.get("main_sheet_name"))
        final_exists = bool(final_result.get("main_sheet_exists", False))

        if not final_name or (not final_exists and all_candidates):
            final_result = _build_forced_fallback_result(
                "Final result had no valid main_sheet_name after synthesis pipeline."
            )

        name = final_result.get("main_sheet_name")
        exists = bool(final_result.get("main_sheet_exists", False))
        confidence = float(final_result.get("confidence", 0.0) or 0.0)
        raw_content = json.dumps(final_result, ensure_ascii=False)

        add_step_log(state, "synthesize:end", {
            "final_name": name,
            "final_exists": exists,
            "confidence": confidence,
            "decision_mode": (
                _parse_json_from_text(raw_content) or {}
            ).get("decision_mode", "unknown"),
        })

        return {
            **state,
            "final_answer": raw_content,
            "main_sheet_name": name,
            "has_main_sheet": exists,
            "next_step": "export",
        }

    except Exception as e:
        add_error_log(state, "synthesize_node", e)

        try:
            evidence_index = _build_sheet_evidence_index(state)
            all_candidates = sorted(evidence_index.keys())

            forced_name, forced_ev = _pick_promat_fallback_candidate(state)

            if not forced_name and state.get("detector_candidate"):
                forced_name = _clean_sheet_name(state.get("detector_candidate"))
                forced_ev = _extract_nn_evidence(state, forced_name)

            if not forced_name and all_candidates:
                forced_name = all_candidates[0]
                forced_ev = _extract_nn_evidence(state, forced_name)

            if forced_name:
                stype = _classify_sheet_type(
                    forced_name,
                    forced_ev,
                    _sheet_title_from_candidates(state, forced_name),
                )
                dq = _disqualification_class(forced_ev, stype)

                raw_content = json.dumps({
                    "main_sheet_exists": True,
                    "main_sheet_name": forced_name,
                    "technical_main_sheet": forced_name,
                    "business_main_sheet": forced_name,
                    "decision_mode": "forced_fallback_after_exception",
                    "confidence": float(forced_ev.get("confidence", 0.0) or 0.01),
                    "reasoning": (
                        f"synthesize_node encountered an exception "
                        f"({type(e).__name__}: {e}). "
                        f"The system returned the best PROMAT-consistent fallback "
                        f"sheet '{forced_name}'."
                    ),
                    "api_response": {
                        "main_sheet_exists": True,
                        "main_sheet_name": forced_name,
                    },
                    "business_arbitration": {
                        "technical_winner_sheet_type": stype,
                        "business_candidate": forced_name,
                        "business_candidate_sheet_type": stype,
                        "business_candidate_blocked_by": forced_ev.get("blocked_by"),
                        "business_candidate_disqualification_class": dq,
                        "override_applied": False,
                    },
                    "layer6_court_status": "not_run_due_to_exception",
                    "fallback_used": True,
                    "fallback_reason": str(e),
                }, ensure_ascii=False)

                return {
                    **state,
                    "final_answer": raw_content,
                    "main_sheet_name": forced_name,
                    "has_main_sheet": True,
                    "next_step": "export",
                }
        except Exception as fallback_err:
            add_error_log(state, "synthesize_node:fallback_failure", fallback_err)

        state["error"] = str(e)
        raise


def export_node(state: OrchestratorState) -> OrchestratorState:
    try:
        add_step_log(state, "export:start", {
            "main_sheet_name": state.get("main_sheet_name"),
            "has_main_sheet": state.get("has_main_sheet", False),
            "detector_candidate": state.get("detector_candidate"),
        })
        json_file, md_file = export_artifacts(state)
        add_step_log(state, "export:end", {"json": json_file, "md": md_file})
        return {
            **state,
            "export_file": json_file,
            "md_export_file": md_file,
            "next_step": "done",
        }
    except Exception as e:
        add_error_log(state, "export_node", e)
        state["error"] = str(e)
        raise

def _contains_any(text: str, kws: list[str]) -> bool:
    return any(kw in text for kw in kws)


def _business_signals(sheet_name: str | None, ev: dict, title: str = "") -> dict:
    name_l = _norm_text(sheet_name)
    title_l = _norm_text(title)
    combined = f"{name_l} || {title_l}"

    strong_fs_kws = [
        "balance sheet", "balance sheets",
        "statement of operations", "statement of income",
        "profit and loss", "p&l",
        "cash flow", "cash flows", "statement of cash flows",
        "change in equity", "stockholders' equity",
        "financial statements",
    ]
    weak_report_kws = ["report", "summary"]
    staging_kws = [
        "aje", "adjusting", "adjustments", "elimination",
        "mapping", "bridge", "rollforward", "support", "schedule",
    ]
    source_kws = ["tb", "trial balance", "gl", "ledger", "index"]

    canonical_fs_title_signal = _contains_any(combined, strong_fs_kws)
    weak_report_signal = _contains_any(combined, weak_report_kws)
    staging_sheet_signal = _contains_any(combined, staging_kws)
    source_sheet_signal = _contains_any(combined, source_kws)

    presentation_layout_signal = bool(
        ev.get("COA_SIGNAL") == 1
        and ev.get("HIDDEN_SIGNAL") == 0
        and ev.get("CODE_COLUMN_SIGNAL") == 0
        and ev.get("TB_PATTERN") == 0
        and ev.get("role_in_graph") != "TB"
    )

    final_output_role_signal = bool(
        not staging_sheet_signal
        and not source_sheet_signal
        and presentation_layout_signal
        and (
            canonical_fs_title_signal
            or (weak_report_signal and ev.get("COA_SIGNAL") == 1)
            or ev.get("FS_PATTERN") == 1
        )
    )

    return {
        "CANONICAL_FS_TITLE_SIGNAL": canonical_fs_title_signal,
        "WEAK_REPORT_SIGNAL": weak_report_signal,
        "PRESENTATION_LAYOUT_SIGNAL": presentation_layout_signal,
        "STAGING_SHEET_SIGNAL": staging_sheet_signal,
        "SOURCE_SHEET_NAME_SIGNAL": source_sheet_signal,
        "FINAL_OUTPUT_ROLE_SIGNAL": final_output_role_signal,
    }
    
    
# ─────────────────────────────────────────────────────────────────────────────
#  Routing + graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def route_after_analyze(state):
    return "plan"


def route_after_plan(state):
    return "act"


def route_after_act(state):
    return "court"


def route_after_court(state):
    return "synthesize"


def route_after_synthesize(state):
    return "export"


def build_graph():
    g = StateGraph(OrchestratorState)
    g.add_node("analyze", analyze_node)
    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("court", court_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("export", export_node)

    g.set_entry_point("analyze")
    g.add_conditional_edges("analyze", route_after_analyze, {"plan": "plan"})
    g.add_conditional_edges("plan", route_after_plan, {"act": "act"})
    g.add_conditional_edges("act", route_after_act, {"court": "court"})
    g.add_conditional_edges("court", route_after_court, {"synthesize": "synthesize"})
    g.add_conditional_edges("synthesize", route_after_synthesize, {"export": "export"})
    g.add_edge("export", END)
    return g.compile()