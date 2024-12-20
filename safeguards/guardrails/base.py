from abc import abstractmethod

import weave


class Guardrail(weave.Model):
    """
    The Guardrail class is an abstract base class that extends the weave.Model.

    This class is designed to provide a framework for implementing guardrails
    in the form of the `guard` method. The `guard` method is an abstract method
    that must be implemented by any subclass. It takes a prompt string and
    additional keyword arguments, and returns a list of strings. The specific
    implementation of the `guard` method will define the behavior of the guardrail.

    Attributes:
        None

    Methods:
        guard(prompt: str, **kwargs) -> list[str]:
            Abstract method that must be implemented by subclasses. It takes a
            prompt string and additional keyword arguments, and returns a list
            of strings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    @weave.op()
    def guard(self, prompt: str, **kwargs) -> list[str]:
        pass
