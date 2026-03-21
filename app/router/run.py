from pathlib import Path
import shutil
import tempfile
import json
import traceback
from datetime import datetime
import uuid
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from app.server.orc.graph import build_graph
from app.model.schemas import TaskResponse

router = APIRouter()
graph = build_graph()

logger = logging.getLogger(__name__)


def write_error_debug_file(file_name: str | None, error: Exception) -> str:
    debug_dir = Path("debug_traces")
    debug_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": str(uuid.uuid4()),
        "status": "error",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "input": {
            "file_name": file_name,
        },
        "errors": [
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        ],
    }

    file_path = debug_dir / f"{payload['run_id']}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return str(file_path)


@router.post("/run", response_model=TaskResponse)
def run_task(file: UploadFile = File(...)):
    temp_path = None

    try:
        logger.info("Starting /run for file: %s", file.filename)

        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing file name.")

        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
            raise HTTPException(
                status_code=400,
                detail="Only Excel .xlsx/.xlsm/.xltx/.xltm files are supported.",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        logger.info("Temporary Excel file saved to: %s", temp_path)

        result = graph.invoke({"user_input": temp_path})

        logger.info("Graph completed successfully for file: %s", file.filename)

        return {
            "has_main_sheet": result.get("has_main_sheet", False),
            "main_sheet_name": result.get("main_sheet_name"),
            "json_export_file": result.get("export_file"),
            "md_export_file": result.get("md_export_file"),
        }

    except HTTPException:
        logger.exception("HTTPException while processing file: %s", file.filename)
        raise
    except Exception as e:
        logger.exception("Unhandled exception while processing file: %s", file.filename)

        debug_trace_file = write_error_debug_file(file.filename, e)

        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "json_export_file": debug_trace_file,
            },
        )
    finally:
        file.file.close()
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
                logger.info("Deleted temporary file: %s", temp_path)
            except Exception:
                logger.exception("Failed to delete temporary file: %s", temp_path)