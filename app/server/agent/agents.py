from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI

from app.config.config import get_settings
from app.server.orc.promat import (
    build_research_system_prompt,
    build_coder_system_prompt,
    build_critic_system_prompt,
)
from app.server.orc.promat.court_prompt import (
    build_plaintiff_system_prompt,
    build_defense_system_prompt,
    build_judge_system_prompt,
)
from app.tools.tools import (
    detect_main_sheet,
    inspect_workbook,
    open_excel,
    python_executor,
    read_excel_cell,
    read_excel_row,
    read_excel_column,
    read_excel_table,
    current_time,
)


# ─────────────────────────────────────────────────────────────────────────────
#  LLM factories — each court role has a dedicated temperature
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm(temperature: float) -> AzureChatOpenAI:
    """
    Return an AzureChatOpenAI instance with the specified temperature.

    Court temperatures (enforced by PROMAT):
      Plaintiff → 0.1   strict, factual, low creativity
      Judge     → 0.4   balanced, evidence-based
      Defense   → 0.7   persuasive, but ZERO tolerance for invented facts
    """
    s = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=s.azure_endpoint,
        api_version=s.azure_api_version,
        deployment_name=s.azure_deployment,
        api_key=s.azure_api_key,
        temperature=temperature,
    )


def get_llm() -> AzureChatOpenAI:
    """Default pipeline LLM (temperature=0)."""
    return _make_llm(0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline agents
# ─────────────────────────────────────────────────────────────────────────────

def build_research_agent():
    llm = get_llm()
    return create_agent(
        model=llm,
        tools=[
            detect_main_sheet,
            inspect_workbook,
            open_excel,
            read_excel_cell,
            read_excel_row,
            read_excel_column,
            read_excel_table,
            current_time,
        ],
        system_prompt=build_research_system_prompt(),
        name="research_agent",
    )


def build_coder_agent():
    llm = get_llm()
    return create_agent(
        model=llm,
        tools=[python_executor],
        system_prompt=build_coder_system_prompt(),
        name="coder_agent",
    )


def build_critic_agent():
    llm = get_llm()
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=build_critic_system_prompt(),
        name="critic_agent",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Court agents
#
#  Each court agent is a plain LLM (no tools) — the court operates purely
#  on the text of prior agent outputs and the Hebrew PROMAT rules.
#
#  Temperature assignment:
#    Plaintiff  → 0.1   (temperature enforced by PROMAT)
#    Judge      → 0.4   (temperature enforced by PROMAT)
#    Defense    → 0.7   (temperature enforced by PROMAT, no lying allowed)
# ─────────────────────────────────────────────────────────────────────────────

def build_plaintiff_agent():
    """
    Plaintiff LLM — temperature 0.1.
    Strict, factual reviewer that finds every flaw in agent output.
    Must never invent facts.
    """
    llm = _make_llm(0.1)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=build_plaintiff_system_prompt(),
        name="plaintiff_agent",
    )


def build_defense_agent():
    """
    Defense Attorney LLM — temperature 0.7.
    Argues in favour of the agent's output, but is FORBIDDEN from
    inventing information.  Concedes valid charges and proposes fixes.
    """
    llm = _make_llm(0.7)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=build_defense_system_prompt(),
        name="defense_agent",
    )


def build_judge_agent():
    """
    Judge LLM — temperature 0.4.
    Listens to plaintiff and defense, issues a binding verdict based
    on Hebrew PROMAT rules and direct evidence only.
    Must never invent facts.
    """
    llm = _make_llm(0.4)
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=build_judge_system_prompt(),
        name="judge_agent",
    )