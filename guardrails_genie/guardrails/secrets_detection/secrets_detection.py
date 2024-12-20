import hashlib
import json
import os
import pathlib
import tempfile
from enum import Enum
from typing import Any, Optional

import weave
from pydantic import BaseModel, PrivateAttr

from guardrails_genie.guardrails.base import Guardrail

try:
    import hyperscan
    from detect_secrets import SecretsCollection
    from detect_secrets.settings import default_settings
except ImportError:
    raise ImportError(
        "The `detect-secrets` and the `hyperscan` packages are required for using the SecretsGuardrail. "
        "Please install then by running `pip install detect-secrets hyperscan`."
    )


class REDACTION(str, Enum):
    """
    Enum for different types of redaction modes.
    """

    REDACT_PARTIAL = "REDACT_PARTIAL"
    REDACT_ALL = "REDACT_ALL"
    REDACT_HASH = "REDACT_HASH"
    REDACT_NONE = "REDACT_NONE"


def redact_value(value: str, mode: str) -> str:
    """
    Redacts the given value based on the specified redaction mode.

    Args:
        value (str): The string value to be redacted.
        mode (str): The redaction mode to be applied. It can be one of the following:
            - REDACTION.REDACT_PARTIAL: Partially redacts the value.
            - REDACTION.REDACT_ALL: Fully redacts the value.
            - REDACTION.REDACT_HASH: Redacts the value by hashing it.
            - REDACTION.REDACT_NONE: No redaction is applied.

    Returns:
        str: The redacted value based on the specified mode.
    """
    replacement = value
    if mode == REDACTION.REDACT_PARTIAL:
        replacement = "[REDACTED:]" + value[:2] + ".." + value[-2:] + "[:REDACTED]"
    elif mode == REDACTION.REDACT_ALL:
        replacement = "[REDACTED:]" + ("*" * len(value)) + "[:REDACTED]"
    elif mode == REDACTION.REDACT_HASH:
        replacement = (
            "[REDACTED:]" + hashlib.md5(value.encode()).hexdigest() + "[:REDACTED]"
        )
    return replacement


class SecretsDetectionSimpleResponse(BaseModel):
    """
    A simple response model for secrets detection.

    Attributes:
        contains_secrets (bool): Indicates if secrets were detected.
        explanation (str): Explanation of the detection result.
        redacted_text (Optional[str]): The redacted text if secrets were found.
        risk_score (float): The risk score of the detection result. (0.0, 0.5, 1.0)
    """

    contains_secrets: bool
    explanation: str
    redacted_text: Optional[str] = None
    risk_score: float = 0.0

    @property
    def safe(self) -> bool:
        """
        Property to check if the text is safe (no secrets detected).

        Returns:
            bool: True if no secrets were detected, False otherwise.
        """
        return not self.contains_secrets


class SecretsDetectionResponse(SecretsDetectionSimpleResponse):
    """
    A detailed response model for secrets detection.

    Attributes:
        detected_secrets (dict[str, list[str]]): Dictionary of detected secrets.
    """

    detected_secrets: dict[str, Any] | None = None


class SecretsInfo(BaseModel):
    """
    Model representing information about a detected secret.

    Attributes:
        secret (str): The detected secret value.
        line_number (int): The line number where the secret was found.
    """

    secret: str
    line_number: int


class ScanResult(BaseModel):
    """
    Model representing the result of a secrets scan.

    Attributes:
        detected_secrets (dict[str, Any] | None): Dictionary of detected secrets, or None if no secrets were found.
        modified_prompt (str): The modified prompt with secrets redacted.
        has_secret (bool): Indicates if any secrets were detected.
        risk_score (float): The risk score of the detection result.
    """

    detected_secrets: dict[str, Any] | None = None
    modified_prompt: str
    has_secret: bool
    risk_score: float


class DetectSecretsModel(weave.Model):
    """
    Model for detecting secrets using the detect-secrets library.
    """

    @staticmethod
    def scan(text: str) -> dict[str, list[SecretsInfo]]:
        """
        Scans the given text for secrets using the detect-secrets library.

        Args:
            text (str): The text to scan for secrets.

        Returns:
            dict[str, list[SecretsInfo]]: A dictionary where the keys are secret types and the values are lists of SecretsInfo objects.
        """
        secrets = SecretsCollection()
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(text.encode("utf-8"))
        temp_file.close()

        with default_settings():
            secrets.scan_file(str(temp_file.name))

        unique_secrets = {}
        for file in secrets.files:
            for found_secret in secrets[file]:
                if found_secret.secret_value is None:
                    continue

                secret_type = found_secret.type
                actual_secret = found_secret.secret_value
                line_number = found_secret.line_number

                if secret_type not in unique_secrets:
                    unique_secrets[secret_type] = []

                unique_secrets[secret_type].append(
                    SecretsInfo(secret=actual_secret, line_number=line_number)
                )

        os.remove(temp_file.name)
        return unique_secrets

    @weave.op
    def invoke(self, text: str) -> dict[str, list[SecretsInfo]]:
        """
        Invokes the scan method to detect secrets in the given text.

        Args:
            text (str): The text to scan for secrets.

        Returns:
            dict[str, list[SecretsInfo]]: A dictionary where the keys are secret types and the values are lists of SecretsInfo objects.
        """
        return self.scan(text)


class HyperScanModel(weave.Model):
    """
    Model for detecting secrets using the Hyperscan library.
    We use the Hyperscan library to scan for secrets using regex patterns.
    The patterns are mined from https://github.com/mazen160/secrets-patterns-db
    This model is used in conjunction with the DetectSecretsModel to improve the detection of secrets.
    """

    _db: Any = PrivateAttr()
    _pattern_map: dict[str, str] = PrivateAttr()
    only_high_confidence: bool = False
    ids: list[str] = []

    def _load_patterns(self) -> dict[str, str]:
        """
        Loads the patterns from a JSONL file.

        Returns:
            dict[str, str]: A dictionary where the keys are pattern names and the values are regex patterns.
        """
        patterns = (
            pathlib.Path(__file__).parent.resolve() / "secrets_patterns.jsonl"
        ).open()
        patterns_list = [json.loads(line) for line in patterns]
        if self.only_high_confidence:
            patterns_list = [
                pattern for pattern in patterns_list if pattern["confidence"] == "high"
            ]
        return {pattern["name"]: pattern["regex"] for pattern in patterns_list}

    def __init__(self, **kwargs: Any):
        """
        Initializes the HyperScanModel instance.
        """
        super().__init__(**kwargs)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization method to load patterns and compile the Hyperscan database.
        """
        self._pattern_map = self._load_patterns()
        self.ids = list(self._pattern_map.keys())
        expressions = [pattern.encode() for pattern in self._pattern_map.values()]
        self._db = hyperscan.Database()
        self._db.compile(expressions=expressions, ids=list(range(len(expressions))))

    def scan(self, text: str) -> dict[str, list[SecretsInfo]]:
        """
        Scans the given text for secrets using the Hyperscan library.

        Args:
            text (str): The text to scan for secrets.

        Returns:
            dict[str, list[SecretsInfo]]: A dictionary where the keys are secret types and the values are lists of SecretsInfo objects.
        """
        unique_secrets = {}

        def on_match(idx, start, end, flags, context):
            """
            Callback function for handling matches found by Hyperscan.

            Args:
                idx: The index of the matched pattern.
                start: The start position of the match.
                end: The end position of the match.
                flags: The flags associated with the match.
                context: The context provided to the scan method.
            """
            secret = context["text"][start:end]
            line_number = context["line_number"]
            current_match = unique_secrets.setdefault(self.ids[idx], [])

            if not current_match or len(secret) > len(current_match[0].secret):
                unique_secrets[self.ids[idx]] = [
                    SecretsInfo(line_number=line_number, secret=secret)
                ]

        for line_no, line in enumerate(text.splitlines(), start=1):
            self._db.scan(
                line.encode(),
                match_event_handler=on_match,
                context={"text": line, "line_number": line_no},
            )

        return unique_secrets

    @weave.op
    def invoke(self, text: str) -> dict[str, list[SecretsInfo]]:
        """
        Invokes the scan method to detect secrets in the given text.

        Args:
            text (str): The text to scan for secrets.

        Returns:
            dict[str, list[SecretsInfo]]: A dictionary where the keys are secret types and the values are lists of SecretsInfo objects.
        """
        return self.scan(text)


class SecretsDetectionGuardrail(Guardrail):
    """
    Guardrail class for secrets detection using both detect-secrets and Hyperscan models.

    Attributes:
        redaction (REDACTION): The redaction mode to be applied.
        _detect_secrets_model (Any): Instance of the DetectSecretsModel.
        _hyperscan_model (Any): Instance of the HyperScanModel.
    """

    redaction: REDACTION
    _detect_secrets_model: Any = PrivateAttr()
    _hyperscan_model: Any = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization method to initialize the detect-secrets and Hyperscan models.
        """
        self._detect_secrets_model = DetectSecretsModel()
        self._hyperscan_model = HyperScanModel()

    def __init__(
        self,
        redaction: REDACTION = REDACTION.REDACT_ALL,
        **kwargs,
    ):
        """
        Initializes the SecretsDetectionGuardrail instance.

        Args:
            redaction (REDACTION): The redaction mode to be applied. Defaults to REDACTION.REDACT_ALL.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            redaction=redaction,
        )

    def get_modified_value(
        self, unique_secrets: dict[str, Any], lines: list[str]
    ) -> str:
        """
        Redacts the detected secrets in the given lines of text.

        Args:
            unique_secrets (dict[str, Any]): Dictionary of detected secrets.
            lines (list[str]): List of lines of text.

        Returns:
            str: The modified text with secrets redacted.
        """
        for _, secrets_list in unique_secrets.items():
            for secret_info in secrets_list:
                secret = secret_info.secret
                line_number = secret_info.line_number
                lines[line_number - 1] = lines[line_number - 1].replace(
                    secret, redact_value(secret, self.redaction)
                )

        modified_value = "\n".join(lines)
        return modified_value

    def get_scan_result(
        self, unique_secrets: dict[str, list[SecretsInfo]], lines: list[str]
    ) -> ScanResult | None:
        """
        Generates a ScanResult based on the detected secrets.

        Args:
            unique_secrets (dict[str, list[SecretsInfo]]): Dictionary of detected secrets.
            lines (list[str]): List of lines of text.

        Returns:
            ScanResult | None: The scan result if secrets are detected, otherwise None.
        """
        if unique_secrets:
            modified_value = self.get_modified_value(unique_secrets, lines)
            detected_secrets = {
                k: [i.secret for i in v] for k, v in unique_secrets.items()
            }

            return ScanResult(
                **{
                    "detected_secrets": detected_secrets,
                    "modified_prompt": modified_value,
                    "has_secret": True,
                    "risk_score": 1.0,
                }
            )
        return None

    def scan(self, prompt: str) -> ScanResult:
        """
        Scans the given prompt for secrets using both detect-secrets and Hyperscan models.

        Args:
            prompt (str): The text to scan for secrets.

        Returns:
            ScanResult: The scan result with detected secrets and redacted text.
        """
        if prompt.strip() == "":
            return ScanResult(
                **{
                    "detected_secrets": None,
                    "modified_prompt": prompt,
                    "has_secret": False,
                    "risk_score": 0.0,
                }
            )

        unique_secrets = self._detect_secrets_model.invoke(text=prompt)
        results = self.get_scan_result(unique_secrets, prompt.splitlines())
        if results:
            return results

        unique_secrets = self._hyperscan_model.invoke(text=prompt)
        results = self.get_scan_result(unique_secrets, prompt.splitlines())
        if results:
            results.risk_score = 0.5
            return results

        return ScanResult(
            **{
                "detected_secrets": None,
                "modified_prompt": prompt,
                "has_secret": False,
                "risk_score": 0.0,
            }
        )

    @weave.op
    def guard(
        self,
        prompt: str,
        return_detected_secrets: bool = True,
        **kwargs,
    ) -> SecretsDetectionResponse | SecretsDetectionResponse:
        """
        Guards the given prompt by scanning for secrets and optionally returning detected secrets.

        Args:
            prompt (str): The text to scan for secrets.
            return_detected_secrets (bool): Whether to return detected secrets in the response. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            SecretsDetectionResponse | SecretsDetectionSimpleResponse: The response with scan results and redacted text.
        """
        results = self.scan(prompt)

        explanation_parts = []
        if results.has_secret:
            explanation_parts.append("Found the following secrets in the text:")
            for secret_type, matches in results.detected_secrets.items():
                explanation_parts.append(f"- {secret_type}: {len(matches)} instance(s)")
        else:
            explanation_parts.append("No secrets detected in the text.")

        if return_detected_secrets:
            return SecretsDetectionResponse(
                contains_secrets=results.has_secret,
                detected_secrets=results.detected_secrets,
                explanation="\n".join(explanation_parts),
                redacted_text=results.modified_prompt,
                risk_score=results.risk_score,
            )
        else:
            return SecretsDetectionSimpleResponse(
                contains_secrets=not results.has_secret,
                explanation="\n".join(explanation_parts),
                redacted_text=results.modified_prompt,
                risk_score=results.risk_score,
            )
