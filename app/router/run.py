from pathlib import Path
import shutil
import tempfile
import json
import traceback
from datetime import datetime
import uuid
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.server.orc.graph import build_graph
from app.model.schemas import TaskResponse

router = APIRouter()
graph = build_graph()

logger = logging.getLogger(__name__)

EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}


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


def _is_excel_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in EXCEL_SUFFIXES


def _collect_excel_files(folder: Path) -> list[Path]:
    files = [p for p in folder.rglob("*") if _is_excel_file(p)]
    return sorted(files, key=lambda p: str(p).lower())


def _run_graph_for_file(file_path: str) -> dict:
    return graph.invoke({"user_input": file_path})


def _extract_final_json(result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("final_answer")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _build_single_file_response(input_file: str, result: dict[str, Any]) -> dict[str, Any]:
    final_json = _extract_final_json(result)

    return {
        "mode": "single_file",
        "input_file": input_file,
        "main_sheet_exists": result.get("has_main_sheet", False),
        "main_sheet_name": result.get("main_sheet_name"),
        "is_card_sheet": result.get("is_card_sheet", final_json.get("is_card_sheet")),
        "technical_main_sheet": result.get(
            "technical_main_sheet",
            final_json.get("technical_main_sheet"),
        ),
        "presentation_main_sheet": result.get(
            "presentation_main_sheet",
            final_json.get("presentation_main_sheet"),
        ),
        "technical_tb_sheet": result.get(
            "technical_tb_sheet",
            final_json.get("technical_tb_sheet"),
        ),
        "decision_mode": result.get(
            "decision_mode",
            final_json.get("decision_mode"),
        ),
        "relationship": result.get(
            "relationship",
            final_json.get("relationship"),
        ),
        "json_export_file": result.get("export_file"),
        "md_export_file": result.get("md_export_file"),
    }


def _build_folder_file_response(excel_file: Path, result: dict[str, Any]) -> dict[str, Any]:
    final_json = _extract_final_json(result)

    return {
        "file": str(excel_file),
        "main_sheet_exists": result.get("has_main_sheet", False),
        "main_sheet_name": result.get("main_sheet_name"),
        "is_card_sheet": result.get("is_card_sheet", final_json.get("is_card_sheet")),
        "technical_main_sheet": result.get(
            "technical_main_sheet",
            final_json.get("technical_main_sheet"),
        ),
        "presentation_main_sheet": result.get(
            "presentation_main_sheet",
            final_json.get("presentation_main_sheet"),
        ),
        "technical_tb_sheet": result.get(
            "technical_tb_sheet",
            final_json.get("technical_tb_sheet"),
        ),
        "decision_mode": result.get(
            "decision_mode",
            final_json.get("decision_mode"),
        ),
        "relationship": result.get(
            "relationship",
            final_json.get("relationship"),
        ),
        "json_export_file": result.get("export_file"),
        "md_export_file": result.get("md_export_file"),
    }


@router.post("/run")
def run_task(
    file: UploadFile | None = File(default=None),
    folder_path: str | None = Form(default=None),
):
    temp_path = None

    try:
        if file is not None and folder_path:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'file' or 'folder_path', not both.",
            )

        if file is None and not folder_path:
            raise HTTPException(
                status_code=400,
                detail="You must provide either an uploaded Excel file or a server-side folder_path.",
            )

        # ── Single uploaded file mode ─────────────────────────────────────────
        if file is not None:
            logger.info("Starting /run for uploaded file: %s", file.filename)

            if not file.filename:
                raise HTTPException(status_code=400, detail="Missing file name.")

            suffix = Path(file.filename).suffix.lower()
            if suffix not in EXCEL_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail="Only Excel .xlsx/.xlsm/.xltx/.xltm files are supported.",
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name

            logger.info("Temporary Excel file saved to: %s", temp_path)

            result = _run_graph_for_file(temp_path)

            logger.info("Graph completed successfully for uploaded file: %s", file.filename)

            return _build_single_file_response(file.filename, result)

        # ── Folder mode ──────────────────────────────────────────────────────
        folder = Path(folder_path).expanduser()

        logger.info("Starting /run for folder: %s", folder)

        if not folder.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Folder does not exist: {folder}",
            )

        if not folder.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a folder: {folder}",
            )

        excel_files = _collect_excel_files(folder)
        if not excel_files:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No Excel files found in folder: {folder}. "
                    "Supported extensions: .xlsx, .xlsm, .xltx, .xltm"
                ),
            )

        logger.info("Found %d Excel files in folder: %s", len(excel_files), folder)

        results: list[dict] = []

        for excel_file in excel_files:
            logger.info("Running graph for file in folder mode: %s", excel_file)
            try:
                result = _run_graph_for_file(str(excel_file))
                results.append(_build_folder_file_response(excel_file, result))
            except Exception as e:
                logger.exception("Failed processing file in folder mode: %s", excel_file)
                debug_trace_file = write_error_debug_file(str(excel_file), e)
                results.append({
                    "file": str(excel_file),
                    "main_sheet_exists": False,
                    "main_sheet_name": None,
                    "is_card_sheet": None,
                    "technical_main_sheet": None,
                    "presentation_main_sheet": None,
                    "technical_tb_sheet": None,
                    "decision_mode": "no_valid_sheet",
                    "relationship": {
                        "main_to_tb_path": [],
                        "path_valid": False,
                    },
                    "json_export_file": debug_trace_file,
                    "md_export_file": None,
                    "error": str(e),
                })

        logger.info("Folder run completed successfully: %s", folder)

        return {
            "mode": "folder",
            "folder_path": str(folder),
            "file_count": len(excel_files),
            "results": results,
        }

    except HTTPException:
        logger.exception(
            "HTTPException while processing request. file=%s folder_path=%s",
            getattr(file, "filename", None),
            folder_path,
        )
        raise
    except Exception as e:
        logger.exception(
            "Unhandled exception while processing request. file=%s folder_path=%s",
            getattr(file, "filename", None),
            folder_path,
        )

        debug_trace_file = write_error_debug_file(
            getattr(file, "filename", None) or folder_path,
            e,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "json_export_file": debug_trace_file,
            },
        )
    finally:
        if file is not None:
            file.file.close()
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
                logger.info("Deleted temporary file: %s", temp_path)
            except Exception:
                logger.exception("Failed to delete temporary file: %s", temp_path)