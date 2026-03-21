from fastapi import FastAPI
import logging
import warnings

from app.router.health import router as health_router
from app.router.run import router as run_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

warnings.filterwarnings(
    "ignore",
    message=r".*pivotCache.*invalid dependency definitions.*",
    category=UserWarning,
)

app = FastAPI(title="Multi-Agent ORC with LangGraph")

app.include_router(health_router)
app.include_router(run_router)