from pydantic import BaseModel




class TaskResponse(BaseModel):
    has_main_sheet: bool
    main_sheet_name: str | None = None
    json_export_file: str
    md_export_file: str |None = None