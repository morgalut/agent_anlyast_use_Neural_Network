from typing import TypedDict, Dict, Any, Optional


class OrchestratorState(TypedDict, total=False):
    user_input: str
    excel_summary: Dict[str, Any]
    main_sheet_result: Dict[str, Any]

    analysis: str
    plan: str
    tasks: list[dict[str, Any]]
    task_results: list[dict[str, Any]]
    final_answer: str

    main_sheet_name: str | None
    has_main_sheet: bool

    # New JSON fields
    is_card_sheet: str | None
    technical_main_sheet: str | None
    presentation_main_sheet: str | None
    technical_tb_sheet: str | None
    decision_mode: str | None
    relationship: Dict[str, Any]

    export_file: str
    md_export_file: str
    next_step: str
    error: Optional[str]

    debug_trace: Dict[str, Any]
    debug_trace_file: str