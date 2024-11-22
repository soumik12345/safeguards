from typing import Optional, Union

import weave
from openai import OpenAI
from openai.types.chat import ChatCompletion


class OpenAIModel(weave.Model):
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
