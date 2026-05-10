


class LLMFactory:
    def __init__(self, llm_type: str, **kwargs):
        self.llm_type = llm_type
        self.kwargs = kwargs
        if llm_type == "llama_cpp":
            from lingofluent.llm.llama_cpp_llm import LlamaCppLLM
            self.llm = LlamaCppLLM(**kwargs)
        elif llm_type == "openai":
            from lingofluent.llm.openai_llm import OpenAILLM
            self.llm = OpenAILLM(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    