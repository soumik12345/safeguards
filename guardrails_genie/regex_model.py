from typing import List, Dict, Optional
import re
import weave
from pydantic import BaseModel


class RegexResult(BaseModel):
    passed: bool
    matched_patterns: Dict[str, List[str]]
    failed_patterns: List[str]


class RegexModel(weave.Model):
    patterns: Dict[str, str]

    def __init__(self, patterns: Dict[str, str]) -> None:
        """
        Initialize RegexModel with a dictionary of patterns.
        
        Args:
            patterns: Dictionary where key is pattern name and value is regex pattern
                     Example: {"email": r"[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+",
                              "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"}
        """
        super().__init__(patterns=patterns)
        self._compiled_patterns = {
            name: re.compile(pattern) for name, pattern in patterns.items()
        }

    @weave.op()
    def check(self, prompt: str) -> RegexResult:
        """
        Check text against all patterns and return detailed results.
        
        Args:
            text: Input text to check against patterns
            
        Returns:
            RegexResult containing pass/fail status and details about matches
        """
        matched_patterns = {}
        failed_patterns = []
        
        for pattern_name, pattern in self.patterns.items():
            matches = []
            for match in re.finditer(pattern, prompt):
                if match.groups():
                    # If there are capture groups, join them with a separator
                    matches.append('-'.join(str(g) for g in match.groups() if g is not None))
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
            passed=len(matched_patterns) == 0
        )

    @weave.op()
    def predict(self, text: str) -> RegexResult:
        """
        Alias for check() to maintain consistency with other models.
        """
        return self.check(text)