import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """Deterministic LLM stub so tests never call OpenAI."""

    available = True

    def with_structured_output(self, schema):
        stub = MockLLM()
        stub._schema = schema
        return stub

    def invoke(self, prompt, *args, **kwargs):
        schema = getattr(self, "_schema", None)
        if schema is not None:
            schema_name = getattr(schema, "__name__", "")
            if schema_name == "FindingResult":
                return agent.FindingResult(findings=[])
            if schema_name == "FileClassificationResult":
                return agent.FileClassificationResult(files=[])
        return MockLLMResponse("Automated PR summary for testing.")


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    stub = MockLLM()
    monkeypatch.setattr(agent, "llm", stub)
    monkeypatch.setattr(agent, "create_llm", lambda *args, **kwargs: stub)

    def llm_for_model(model_name=None):
        return stub

    monkeypatch.setattr(agent, "llm_for_model", llm_for_model)
