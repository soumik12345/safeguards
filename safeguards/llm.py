from typing import Optional, Union

import weave
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion


class OpenAIModel(weave.Model):
    """
    A class to interface with OpenAI's language models using the Weave framework.

    This class provides methods to create structured messages and generate predictions
    using OpenAI's chat completion API. It is designed to work with both single and
    multiple user prompts, and optionally includes a system prompt to guide the model's
    responses.

    Args:
        model_name (str): The name of the OpenAI model to be used for predictions.
    """

    model_name: str
    _openai_client: OpenAI

    def __init__(self, model_name: str = "gpt-4o") -> None:
        super().__init__(model_name=model_name)
        self._openai_client = OpenAI()

    @weave.op()
    def create_messages(
        self,
        user_prompts: Union[str, list[str]],
        system_prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Create a list of messages for the OpenAI chat completion API.

        This function constructs a list of messages in the format required by the
        OpenAI chat completion API. It takes user prompts, an optional system prompt,
        and an optional list of existing messages, and combines them into a single
        list of messages.

        Args:
            user_prompts (Union[str, list[str]]): A single user prompt or a list of
                user prompts to be included in the messages.
            system_prompt (Optional[str]): An optional system prompt to guide the
                model's responses. If provided, it will be added at the beginning
                of the messages list.
            messages (Optional[list[dict]]): An optional list of existing messages
                to which the new prompts will be appended. If not provided, a new
                list will be created.

        Returns:
            list[dict]: A list of messages formatted for the OpenAI chat completion API.
        """
        user_prompts = [user_prompts] if isinstance(user_prompts, str) else user_prompts
        messages = list(messages) if isinstance(messages, dict) else []
        for user_prompt in user_prompts:
            messages.append({"role": "user", "content": user_prompt})
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + messages
        return messages

    @weave.op()
    def predict(
        self,
        user_prompts: Union[str, list[str]],
        system_prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        **kwargs,
    ) -> ChatCompletion:
        """
        Generate a chat completion response using the OpenAI API.

        This function takes user prompts, an optional system prompt, and an optional
        list of existing messages to create a list of messages formatted for the
        OpenAI chat completion API. It then sends these messages to the OpenAI API
        to generate a chat completion response.

        Args:
            user_prompts (Union[str, list[str]]): A single user prompt or a list of
                user prompts to be included in the messages.
            system_prompt (Optional[str]): An optional system prompt to guide the
                model's responses. If provided, it will be added at the beginning
                of the messages list.
            messages (Optional[list[dict]]): An optional list of existing messages
                to which the new prompts will be appended. If not provided, a new
                list will be created.
            **kwargs: Additional keyword arguments to be passed to the OpenAI API
                for chat completion.

        Returns:
            ChatCompletion: The chat completion response from the OpenAI API.
        """
        messages = self.create_messages(user_prompts, system_prompt, messages)
        if "response_format" in kwargs:
            response = self._openai_client.beta.chat.completions.parse(
                model=self.model_name, messages=messages, **kwargs
            )
        else:
            response = self._openai_client.chat.completions.create(
                model=self.model_name, messages=messages, **kwargs
            )
        return response


class AsyncOpenAIModel(weave.Model):
    """
    A class to interface with OpenAI's language models using the Weave framework.

    This class provides methods to create structured messages and generate predictions
    using AsyncOpenAI's chat completion API. It is designed to work with both single and
    multiple user prompts, and optionally includes a system prompt to guide the model's
    responses.

    Args:
        model_name (str): The name of the OpenAI model to be used for predictions.
    """

    model_name: str
    _openai_client: AsyncOpenAI

    def __init__(self, model_name: str = "gpt-4o") -> None:
        super().__init__(model_name=model_name)
        self._openai_client = AsyncOpenAI()

    @weave.op()
    def create_messages(
        self,
        user_prompts: Union[str, list[str]],
        system_prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Create a list of messages for the OpenAI chat completion API.

        This function constructs a list of messages in the format required by the
        OpenAI chat completion API. It takes user prompts, an optional system prompt,
        and an optional list of existing messages, and combines them into a single
        list of messages.

        Args:
            user_prompts (Union[str, list[str]]): A single user prompt or a list of
                user prompts to be included in the messages.
            system_prompt (Optional[str]): An optional system prompt to guide the
                model's responses. If provided, it will be added at the beginning
                of the messages list.
            messages (Optional[list[dict]]): An optional list of existing messages
                to which the new prompts will be appended. If not provided, a new
                list will be created.

        Returns:
            list[dict]: A list of messages formatted for the OpenAI chat completion API.
        """
        user_prompts = [user_prompts] if isinstance(user_prompts, str) else user_prompts
        messages = list(messages) if isinstance(messages, dict) else []
        for user_prompt in user_prompts:
            messages.append({"role": "user", "content": user_prompt})
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + messages
        return messages

    @weave.op()
    async def predict(
        self,
        user_prompts: Union[str, list[str]],
        system_prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        **kwargs,
    ) -> ChatCompletion:
        """
        Generate a chat completion response using the AsyncOpenAI API.

        This function takes user prompts, an optional system prompt, and an optional
        list of existing messages to create a list of messages formatted for the
        OpenAI chat completion API. It then sends these messages to the OpenAI API
        to generate a chat completion response.

        Args:
            user_prompts (Union[str, list[str]]): A single user prompt or a list of
                user prompts to be included in the messages.
            system_prompt (Optional[str]): An optional system prompt to guide the
                model's responses. If provided, it will be added at the beginning
                of the messages list.
            messages (Optional[list[dict]]): An optional list of existing messages
                to which the new prompts will be appended. If not provided, a new
                list will be created.
            **kwargs: Additional keyword arguments to be passed to the OpenAI API
                for chat completion.

        Returns:
            ChatCompletion: The chat completion response from the OpenAI API.
        """
        messages = self.create_messages(user_prompts, system_prompt, messages)
        if "response_format" in kwargs:
            response = await self._openai_client.beta.chat.completions.parse(
                model=self.model_name, messages=messages, **kwargs
            )
        else:
            response = await self._openai_client.chat.completions.create(
                model=self.model_name, messages=messages, **kwargs
            )
        return response
