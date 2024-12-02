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
    def check(self, text: str) -> RegexResult:
        """
        Check text against all patterns and return detailed results.
        
        Args:
            text: Input text to check against patterns
            
        Returns:
            RegexResult containing pass/fail status and details about matches
        """
        matches: Dict[str, List[str]] = {}
        failed_patterns: List[str] = []
        
        for pattern_name, compiled_pattern in self._compiled_patterns.items():
            found_matches = compiled_pattern.findall(text)
            if found_matches:
                matches[pattern_name] = found_matches
            else:
                failed_patterns.append(pattern_name)
        
        # Consider it passed only if no patterns matched (no PII found)
        passed = len(matches) == 0
        
        return RegexResult(
            passed=passed,
            matched_patterns=matches,
            failed_patterns=failed_patterns
        )

    @weave.op()
    def predict(self, text: str) -> RegexResult:
        """
        Alias for check() to maintain consistency with other models.
        """
        return self.check(text)