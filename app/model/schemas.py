from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class TaskResponse(BaseModel):
    has_main_sheet: bool
    main_sheet_name: Optional[str] = None

    # TB (card sheet) — string or None
    is_card_sheet: Optional[str] = None

    # Dual-truth fields
    technical_main_sheet: Optional[str] = None
    presentation_main_sheet: Optional[str] = None
    technical_tb_sheet: Optional[str] = None

    decision_mode: Optional[str] = None

    # Relationship graph info
    relationship: Optional[Dict[str, Any]] = None

    json_export_file: str
    md_export_file: Optional[str] = None