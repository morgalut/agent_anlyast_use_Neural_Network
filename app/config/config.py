from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseModel):
    azure_endpoint: str
    azure_api_version: str
    azure_deployment: str
    azure_api_key: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        azure_endpoint=os.environ["AZURE_OPENAI_GPT54_ENDPOINT"],
        azure_api_version=os.environ["AZURE_OPENAI_GPT54_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_GPT54_DEPLOYMENT_NAME"],
        azure_api_key=os.environ["AZURE_OPENAI_GPT54_API_KEY"],
    )