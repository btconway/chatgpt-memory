from chatgpt_memory.llm_client.config import LLMClientConfig


class ChatGPTConfig(LLMClientConfig):
    temperature: float = 0
    model_name: str = "gpt-3.5-turbo-16k-0613"
    max_retries: int = 6
    max_tokens: int = 12000
    verbose: bool = False
