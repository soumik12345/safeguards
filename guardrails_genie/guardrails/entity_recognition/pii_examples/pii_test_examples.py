"""
Collection of PII test examples with expected outcomes for entity recognition testing.
Each example includes the input text and expected entities to be detected.
"""

PII_TEST_EXAMPLES = [
    {
        "description": "Business Context - Employee Record",
        "input_text": """
Please update our records for employee John Smith:
Email: john.smith@company.com
Phone: 123-456-7890
SSN: 123-45-6789
Emergency Contact: Mary Johnson (Tel: 098-765-4321)
""",
        "expected_entities": {
            "GIVENNAME": ["John", "Mary"],
            "SURNAME": ["Smith", "Johnson"],
            "EMAIL": ["john.smith@company.com"],
            "PHONE_NUMBER": ["123-456-7890", "098-765-4321"],
            "SOCIALNUM": ["123-45-6789"],
        },
    },
    {
        "description": "Meeting Notes with Attendees",
        "input_text": """
Meeting Notes - Project Alpha
Date: 2024-03-15
Attendees: 
- Sarah Williams (sarah.w@company.com)
- Robert Brown (bobby@email.com)
- Tom Wilson (555-0123-4567)

Action Items:
1. Sarah to review documentation
2. Contact Bob at his alternate number: 777-888-9999
""",
        "expected_entities": {
            "GIVENNAME": ["Sarah", "Robert", "Tom", "Bob"],
            "SURNAME": ["Williams", "Brown", "Wilson"],
            "EMAIL": ["sarah.w@company.com", "bobby@email.com"],
            "PHONE_NUMBER": ["555-0123-4567", "777-888-9999"],
        },
    },
    {
        "description": "Medical Record",
        "input_text": """
Patient: Emma Thompson
DOB: 05/15/1980
Medical Record #: MR-12345
Primary Care: Dr. James Wilson
Contact: emma.t@email.com
Insurance ID: INS-987654321
Emergency Contact: Michael Thompson (555-123-4567)
""",
        "expected_entities": {
            "GIVENNAME": ["Emma", "James", "Michael"],
            "SURNAME": ["Thompson", "Wilson", "Thompson"],
            "EMAIL": ["emma.t@email.com"],
            "PHONE_NUMBER": ["555-123-4567"],
        },
    },
    {
        "description": "No PII Content",
        "input_text": """
Project Status Update:
- All deliverables are on track
- Budget is within limits
- Next review scheduled for next week
""",
        "expected_entities": {},
    },
    {
        "description": "Mixed Format Phone Numbers",
        "input_text": """
Contact Directory:
Main Office: (555) 123-4567
Support: 555.987.6543
International: +1-555-321-7890
Emergency: 555 444 3333
""",
        "expected_entities": {
            "PHONE_NUMBER": [
                "(555) 123-4567",
                "555.987.6543",
                "+1-555-321-7890",
                "555 444 3333",
            ]
        },
    },
]

# Additional examples can be added to test specific edge cases or formats
EDGE_CASE_EXAMPLES = [
    {
        "description": "Mixed Case and Special Characters",
        "input_text": """
JoHn.DoE@company.com
JANE_SMITH@email.com
bob.jones123@domain.co.uk
""",
        "expected_entities": {
            "EMAIL": [
                "JoHn.DoE@company.com",
                "JANE_SMITH@email.com",
                "bob.jones123@domain.co.uk",
            ],
            "GIVENNAME": ["John", "Jane", "Bob"],
            "SURNAME": ["Doe", "Smith", "Jones"],
        },
    }
]


def validate_entities(detected: dict, expected: dict) -> bool:
    """Compare detected entities with expected entities"""
    if set(detected.keys()) != set(expected.keys()):
        return False
    return all(set(detected[k]) == set(expected[k]) for k in expected.keys())


def run_test_case(guardrail, test_case, test_type="Main"):
    """Run a single test case and print results"""
    print(f"\n{test_type} Test Case: {test_case['description']}")
    print("-" * 50)

    result = guardrail.guard(test_case["input_text"])
    expected = test_case["expected_entities"]

    # Validate results
    matches = validate_entities(result.detected_entities, expected)

    print(f"Test Status: {'✓ PASS' if matches else '✗ FAIL'}")
    print(f"Contains PII: {result.contains_entities}")

    if not matches:
        print("\nEntity Comparison:")
        all_entity_types = set(
            list(result.detected_entities.keys()) + list(expected.keys())
        )
        for entity_type in all_entity_types:
            detected = set(result.detected_entities.get(entity_type, []))
            expected_set = set(expected.get(entity_type, []))
            print(f"\nEntity Type: {entity_type}")
            print(f"  Expected: {sorted(expected_set)}")
            print(f"  Detected: {sorted(detected)}")
            if detected != expected_set:
                print(f"  Missing: {sorted(expected_set - detected)}")
                print(f"  Extra: {sorted(detected - expected_set)}")

    if result.anonymized_text:
        print(f"\nAnonymized Text:\n{result.anonymized_text}")

    return matches
