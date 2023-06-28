"""Utils for using OpenAI API"""
import json
import logging
from typing import Any, Dict, Tuple, Union

import requests
from transformers import GPT2TokenizerFast

from chatgpt_memory.environment import OPENAI_BACKOFF, OPENAI_MAX_RETRIES, OPENAI_TIMEOUT
from chatgpt_memory.errors import OpenAIError, OpenAIRateLimitError
from chatgpt_memory.utils.reflection import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


def load_openai_tokenizer(tokenizer_name: str, use_tiktoken: bool) -> Any:
    """
    Load either the tokenizer from tiktoken (if the library is available) or
    fallback to the GPT2TokenizerFast from the transformers library.

    Args:
        tokenizer_name (str): The name of the tokenizer to load.
        use_tiktoken (bool): Use tiktoken tokenizer or not.

    Raises:
        ImportError: When `tiktoken` package is missing.
        To use tiktoken tokenizer install it as follows:
        `pip install tiktoken`

    Returns:
        tokenizer: Tokenizer of either GPT2 kind or tiktoken based.
    """
    tokenizer = None
    if use_tiktoken:
        try:
            import tiktoken  # pylint: disable=import-error

            logger.debug("Using tiktoken %s tokenizer", tokenizer_name)
            tokenizer = tiktoken.get_encoding(tokenizer_name)
        except ImportError:
            raise ImportError(
                "The `tiktoken` package not found.",
                "To install it use the following:",
                "`pip install tiktoken`",
            )
    else:
        logger.warning(
            "OpenAI tiktoken module is not available for Python < 3.8,Linux ARM64 and "
            "AARCH64. Falling back to GPT2TokenizerFast."
        )

        logger.debug("Using GPT2TokenizerFast tokenizer")
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    return tokenizer


def count_openai_tokens(text: str, tokenizer: Any, use_tiktoken: bool) -> int:
    """
    Count the number of tokens in `text` based on the provided OpenAI `tokenizer`.

    Args:
        text (str):  A string to be tokenized.
        tokenizer (Any): An OpenAI tokenizer.
        use_tiktoken (bool): Use tiktoken tokenizer or not.

    Returns:
        int: Number of tokens in the text.
    """

    if use_tiktoken:
        return len(tokenizer.encode(text))
    else:
        return len(tokenizer.tokenize(text))


@retry_with_exponential_backoff(
    backoff_in_seconds=OPENAI_BACKOFF,
    max_retries=OPENAI_MAX_RETRIES,
    errors=(OpenAIRateLimitError, OpenAIError),
)
def openai_request(
    url: str,
    headers: Dict,
    payload: Dict,
    timeout: Union[float, Tuple[float, float]] = OPENAI_TIMEOUT,
) -> Dict:
    """
    Make a request to the OpenAI API given a `url`, `headers`, `payload`, and
    `timeout`.

    Args:
        url (str): The URL of the OpenAI API.
        headers (Dict): Dictionary of HTTP Headers to send with the :class:`Request`.
        payload (Dict): The payload to send with the request.
        timeout (Union[float, Tuple[float, float]], optional): The timeout length of the request. The default is 30s.
        Defaults to OPENAI_TIMEOUT.

    Raises:
        openai_error: If the request fails.

    Returns:
        Dict: OpenAI Embedding API response.
    """

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload), timeout=timeout)
    res = json.loads(response.text)

    # if request is unsucessful and `status_code = 429` then,
    # raise rate limiting error else the OpenAIError
    if response.status_code != 200:
        openai_error: OpenAIError
        if response.status_code == 429:
            openai_error = OpenAIRateLimitError(f"API rate limit exceeded: {response.text}")
        else:
            openai_error = OpenAIError(
                f"OpenAI returned an error.\n"
                f"Status code: {response.status_code}\n"
                f"Response body: {response.text}",
                status_code=response.status_code,
            )
        raise openai_error

    return res


def get_prompt(message: str, history: str) -> str:
    """
    Generates the prompt based on the current history and message.

    Args:
        message (str): Current message from user.
        history (str): Retrieved history for the current message.
        History follows the following format for example:
        ```
        Human: hello
        Assistant: hello, how are you?
        Human: good, you?
        Assistant: I am doing good as well. How may I help you?
        ```
    Returns:
        prompt: Curated prompt for the ChatGPT API based on current params.
    """
    prompt = f""" You are an AI Assistant designed to intelligently generate content for a wide variety of tasks, with a focus on sales and marketing materials. Your main task is to provide users with high-quality content based on your extensive knowledge base and understanding of various writing styles.

Incorporate Context: When given context or reference material, utilize it effectively to generate content that aligns with the user's needs. Pay special attention to any context that provides examples of preferred writing style and tone. For instance, if you are asked to generate content related to VNTANA, remember that it is a 3D Infrastructure Platform designed to optimize, convert, and distribute 3D assets for use in various channels. The platform is used by brands, retailers, and technology platforms to create immersive 3D and AR experiences, streamline 3D workflows, and integrate 3D capabilities into existing infrastructure.

Adapt to Writing Styles: Be flexible in adopting different writing styles as desired by the user, such as emulating a specific person's style or following principles for selling and persuasion. Ensure that your generated content maintains the desired tone, format, and structure.

Core Message and Key Points: Keep in mind the core message and key points you need to convey in your generated content. Focus on addressing the user's main objectives and make sure the content is aligned with their goals. For example, if the user's goal is to highlight the benefits of VNTANA's platform, emphasize its ability to automatically optimize 3D models, connect all tools in a 3D workflow, and make 3D accessible to every part of a business.

Wide Variety of Tasks: Be prepared to assist users with various tasks, ranging from email replies to creating sales and marketing materials. Use your knowledge base to provide targeted and relevant content for each specific task.

Generate Content Efficiently: Produce high-quality content that is concise, informative, and impactful. Ensure that your writing is clear, well-structured, and persuasive to help users achieve their desired outcomes.

Remember, the overall purpose is to support users by generating content that effectively addresses their needs and objectives across a wide variety of tasks.

If the AI Assistant does not know the answer to a question, it truthfully says it does not know. The AI Assisaant ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

    {history}
    Human: {message}
    Assistant:"""

    return prompt
