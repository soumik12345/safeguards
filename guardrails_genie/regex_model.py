from typing import Optional, Union

import regex as re
import weave
from pydantic import BaseModel


class RegexResult(BaseModel):
    passed: bool
    matched_patterns: dict[str, list[str]]
    failed_patterns: list[str]


class RegexModel(weave.Model):
    """
    Initialize RegexModel with a dictionary of patterns.

    Args:
        patterns (Dict[str, str]): Dictionary where key is pattern name and value is regex pattern.
    """

    patterns: Optional[Union[dict[str, str], dict[str, list[str]]]] = None

    def __init__(
        self, patterns: Optional[Union[dict[str, str], dict[str, list[str]]]] = None
    ) -> None:
        super().__init__(patterns=patterns)
        normalized_patterns = {}
        for k, v in patterns.items():
            normalized_patterns[k] = v if isinstance(v, list) else [v]
        self._compiled_patterns = {
            name: [re.compile(p) for p in pattern]
            for name, pattern in normalized_patterns.items()
        }

    @weave.op()
    def check(self, text: str) -> RegexResult:
        """
        Check text against all patterns and return detailed results.

        Args:
            text: Input text to check against patterns

        Returns:
            RegexResult containing pass/fail status and details about matches
        """
        matched_patterns = {}
        failed_patterns = []

        for pattern_name, pats in self._compiled_patterns.items():
            matches = []
            for pattern in pats:
                for match in pattern.finditer(text):
                    if match.groups():
                        # If there are capture groups, join them with a separator
                        matches.append(
                            "-".join(str(g) for g in match.groups() if g is not None)
                        )
                    else:
                        # If no capture groups, use the full match
                        matches.append(match.group(0))

            if matches:
                matched_patterns[pattern_name] = matches
            else:
                failed_patterns.append(pattern_name)

        return RegexResult(
            matched_patterns=matched_patterns,
            failed_patterns=failed_patterns,
            passed=len(matched_patterns) == 0,
        )

    @weave.op()
    def predict(self, text: str) -> RegexResult:
        """
        Alias for check() to maintain consistency with other models.
        """
        return self.check(text)
