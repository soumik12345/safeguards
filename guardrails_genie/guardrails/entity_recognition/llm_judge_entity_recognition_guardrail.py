from typing import Dict, List, Optional

import instructor
import weave
from pydantic import BaseModel, Field

from ...llm import OpenAIModel
from ..base import Guardrail


class TermMatch(BaseModel):
    """Represents a matched term and its variations"""

    original_term: str
    matched_text: str
    match_type: str = Field(
        description="Type of match: EXACT, MISSPELLING, ABBREVIATION, or VARIANT"
    )
    explanation: str = Field(
        description="Explanation of why this is considered a match"
    )


class RestrictedTermsAnalysis(BaseModel):
    """Analysis result for restricted terms detection"""

    contains_restricted_terms: bool = Field(
        description="Whether any restricted terms were detected"
    )
    detected_matches: List[TermMatch] = Field(
        default_factory=list,
        description="List of detected term matches with their variations",
    )
    explanation: str = Field(description="Detailed explanation of the analysis")
    anonymized_text: Optional[str] = Field(
        default=None,
        description="Text with restricted terms replaced with category tags",
    )

    @property
    def safe(self) -> bool:
        return not self.contains_restricted_terms


class RestrictedTermsRecognitionResponse(BaseModel):
    contains_entities: bool
    detected_entities: Dict[str, List[str]]
    explanation: str
    anonymized_text: Optional[str] = None

    @property
    def safe(self) -> bool:
        return not self.contains_entities


class RestrictedTermsJudge(Guardrail):
    """
    A class to detect and analyze restricted terms and their variations in text using an LLM model.

    The RestrictedTermsJudge class extends the Guardrail class and utilizes an OpenAIModel
    to identify restricted terms and their variations within a given text. It provides
    functionality to format prompts for the LLM, predict restricted terms, and optionally
    anonymize detected terms in the text.

    !!! example "Using RestrictedTermsJudge"
        ```python
        from guardrails_genie.guardrails.entity_recognition import RestrictedTermsJudge

        # Initialize with OpenAI model
        guardrail = RestrictedTermsJudge(should_anonymize=True)

        # Check for specific terms
        result = guardrail.guard(
            text="Let's implement features like Salesforce",
            custom_terms=["Salesforce", "Oracle", "AWS"]
        )
        ```

    Attributes:
        llm_model (OpenAIModel): An instance of OpenAIModel used for predictions.
        should_anonymize (bool): A flag indicating whether detected terms should be anonymized.

    Args:
        should_anonymize (bool): A flag indicating whether detected terms should be anonymized.
    """

    llm_model: OpenAIModel = Field(default_factory=lambda: OpenAIModel())
    should_anonymize: bool = False

    def __init__(self, should_anonymize: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.should_anonymize = should_anonymize
        # Pass the OpenAI client to instructor.patch()
        instructor.patch(client=self.llm_model._openai_client)

    def format_prompts(self, text: str, custom_terms: List[str]) -> tuple[str, str]:
        user_prompt = f"""Analyze the following text for restricted terms and variations:

<text_to_analyze>
{text}
</text_to_analyze>

Look for these specific terms and their variations:
{', '.join(custom_terms)}

Analyze the text carefully for:
1. Exact matches
2. Common misspellings
3. Abbreviations
4. Intentional variations (l33t speak, special characters, etc.)
5. Case variations
"""

        system_prompt = """You are an expert system for detecting restricted terms, brand mentions, and inappropriate content.

Your task is to:
1. Identify exact matches of restricted terms
2. Detect variations including:
   - Misspellings (both accidental and intentional)
   - Abbreviations and acronyms
   - Case variations
   - L33t speak or special character substitutions
   - Partial matches within larger words

For each match, you must:
1. Identify the original restricted term
2. Note the actual text that matched
3. Classify the match type
4. Provide a confidence score
5. Explain why it's considered a match

Be thorough but avoid false positives. Focus on meaningful matches that indicate actual attempts to use restricted terms.

Return your analysis in the structured format specified by the RestrictedTermsAnalysis model."""

        return user_prompt, system_prompt

    @weave.op()
    def predict(
        self, text: str, custom_terms: List[str], **kwargs
    ) -> RestrictedTermsAnalysis:
        user_prompt, system_prompt = self.format_prompts(text, custom_terms)

        response = self.llm_model.predict(
            user_prompts=user_prompt,
            system_prompt=system_prompt,
            response_format=RestrictedTermsAnalysis,
            temperature=0.1,  # Lower temperature for more consistent analysis
            **kwargs,
        )

        return response.choices[0].message.parsed

    # TODO: Remove default custom_terms
    @weave.op()
    def guard(
        self,
        text: str,
        custom_terms: List[str] = [
            "Microsoft",
            "Amazon Web Services",
            "Facebook",
            "Meta",
            "Google",
            "Salesforce",
            "Oracle",
        ],
        aggregate_redaction: bool = True,
        **kwargs,
    ) -> RestrictedTermsRecognitionResponse:
        """
        Analyzes the provided text to identify and handle restricted terms and their variations.

        This function utilizes a predictive model to scan the input text for any occurrences of
        specified restricted terms, including their variations such as misspellings, abbreviations,
        and case differences. It returns a detailed analysis of the findings, including whether
        restricted terms were detected, a summary of the matches, and an optional anonymized version
        of the text.

        The function operates by first calling the `predict` method to perform the analysis based on
        the given text and custom terms. If restricted terms are found, it constructs a summary of
        these findings. Additionally, if anonymization is enabled, it replaces detected terms in the
        text with a redacted placeholder or a specific match type indicator, depending on the
        `aggregate_redaction` flag.

        Args:
            text (str): The text to be analyzed for restricted terms.
            custom_terms (List[str]): A list of restricted terms to check against the text. Defaults
                to a predefined list of company names.
            aggregate_redaction (bool): Determines the anonymization strategy. If True, all matches
                are replaced with "[redacted]". If False, matches are replaced
                with their match type in uppercase.

        Returns:
            RestrictedTermsRecognitionResponse: An object containing the results of the analysis,
                including whether restricted terms were found, a dictionary of detected entities,
                a summary explanation, and the anonymized text if applicable.
        """
        analysis = self.predict(text, custom_terms, **kwargs)

        # Create a summary of findings
        if analysis.contains_restricted_terms:
            summary_parts = ["Restricted terms detected:"]
            for match in analysis.detected_matches:
                summary_parts.append(
                    f"\n- {match.original_term}: {match.matched_text} ({match.match_type})"
                )
            summary = "\n".join(summary_parts)
        else:
            summary = "No restricted terms detected."

        # Updated anonymization logic
        anonymized_text = None
        if self.should_anonymize and analysis.contains_restricted_terms:
            anonymized_text = text
            for match in analysis.detected_matches:
                replacement = (
                    "[redacted]"
                    if aggregate_redaction
                    else f"[{match.match_type.upper()}]"
                )
                anonymized_text = anonymized_text.replace(
                    match.matched_text, replacement
                )

        # Convert detected_matches to a dictionary format
        detected_entities = {}
        for match in analysis.detected_matches:
            if match.original_term not in detected_entities:
                detected_entities[match.original_term] = []
            detected_entities[match.original_term].append(match.matched_text)

        return RestrictedTermsRecognitionResponse(
            contains_entities=analysis.contains_restricted_terms,
            detected_entities=detected_entities,
            explanation=summary,
            anonymized_text=anonymized_text,
        )
