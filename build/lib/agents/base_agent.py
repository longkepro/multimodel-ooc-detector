from src.llm_provider import llm_provider

class BaseAgent:
    def __init__(self):
        self.llm = llm_provider