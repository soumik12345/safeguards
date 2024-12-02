"""
Collection of restricted terms test examples with expected outcomes for entity recognition testing.
Focuses on banned terms, competitor mentions, and brand protection scenarios.
"""

RESTRICTED_TERMS_EXAMPLES = [
    {
        "description": "Competitor Product Discussion",
        "input_text": """
I think we should implement features similar to Salesforce's Einstein AI 
and Oracle's Cloud Infrastructure. Maybe we could also look at how 
AWS handles their lambda functions.
""",
        "custom_terms": ["Salesforce", "Oracle", "AWS", "Einstein AI", "Cloud Infrastructure", "lambda"],
        "expected_entities": {
            "Salesforce": ["Salesforce"],
            "Oracle": ["Oracle"],
            "AWS": ["AWS"],
            "Einstein AI": ["Einstein AI"],
            "Cloud Infrastructure": ["Cloud Infrastructure"],
            "lambda": ["lambda"]
        }
    },
    {
        "description": "Inappropriate Language in Support Ticket",
        "input_text": """
This damn product keeps crashing! What the hell is wrong with your 
stupid service? I've wasted so much freaking time on this crap.
""",
        "custom_terms": ["damn", "hell", "stupid", "crap"],
        "expected_entities": {
            "damn": ["damn"],
            "hell": ["hell"],
            "stupid": ["stupid"],
            "crap": ["crap"]
        }
    },
    {
        "description": "Confidential Project Names",
        "input_text": """
Project Titan's launch date has been moved up. We should coordinate 
with Project Phoenix team and the Blue Dragon initiative for resource allocation.
""",
        "custom_terms": ["Project Titan", "Project Phoenix", "Blue Dragon"],
        "expected_entities": {
            "Project Titan": ["Project Titan"],
            "Project Phoenix": ["Project Phoenix"],
            "Blue Dragon": ["Blue Dragon"]
        }
    }
]

# Edge cases and special formats
EDGE_CASE_EXAMPLES = [
    {
        "description": "Common Corporate Abbreviations and Stock Symbols",
        "input_text": """
MSFT's Azure and O365 platform is gaining market share. 
Have you seen what GOOGL/GOOG and FB/META are doing with their AI?
CRM (Salesforce) and ORCL (Oracle) have interesting features too.
""",
        "custom_terms": ["Microsoft", "Google", "Meta", "Facebook", "Salesforce", "Oracle"],
        "expected_entities": {
            "Microsoft": ["MSFT"],
            "Google": ["GOOGL", "GOOG"],
            "Meta": ["META"],
            "Facebook": ["FB"],
            "Salesforce": ["CRM", "Salesforce"],
            "Oracle": ["ORCL"]
        }
    },
    {
        "description": "L33t Speak and Intentional Obfuscation",
        "input_text": """
S4l3sf0rc3 is better than 0r4cl3! 
M1cr0$oft and G00gl3 are the main competitors.
Let's check F8book and Met@ too.
""",
        "custom_terms": ["Salesforce", "Oracle", "Microsoft", "Google", "Facebook", "Meta"],
        "expected_entities": {
            "Salesforce": ["S4l3sf0rc3"],
            "Oracle": ["0r4cl3"],
            "Microsoft": ["M1cr0$oft"],
            "Google": ["G00gl3"],
            "Facebook": ["F8book"],
            "Meta": ["Met@"]
        }
    },
    {
        "description": "Case Variations and Partial Matches",
        "input_text": """
salesFORCE and ORACLE are competitors.
MicroSoft and google are too.
Have you tried micro-soft or Google_Cloud?
""",
        "custom_terms": ["Microsoft", "Google", "Salesforce", "Oracle"],
        "expected_entities": {
            "Microsoft": ["MicroSoft", "micro-soft"],
            "Google": ["google", "Google_Cloud"],
            "Salesforce": ["salesFORCE"],
            "Oracle": ["ORACLE"]
        }
    },
    {
        "description": "Common Misspellings and Typos",
        "input_text": """
Microsft and Microsooft are common typos.
Goggle, Googel, and Gooogle are search related.
Salezforce and Oracel need checking too.
""",
        "custom_terms": ["Microsoft", "Google", "Salesforce", "Oracle"],
        "expected_entities": {
            "Microsoft": ["Microsft", "Microsooft"],
            "Google": ["Goggle", "Googel", "Gooogle"],
            "Salesforce": ["Salezforce"],
            "Oracle": ["Oracel"]
        }
    },
    {
        "description": "Mixed Variations and Context",
        "input_text": """
The M$ cloud competes with AWS (Amazon Web Services).
FB/Meta's social platform and GOOGL's search dominate.
SF.com and Oracle-DB are industry standards.
""",
        "custom_terms": ["Microsoft", "Amazon Web Services", "Facebook", "Meta", "Google", "Salesforce", "Oracle"],
        "expected_entities": {
            "Microsoft": ["M$"],
            "Amazon Web Services": ["AWS"],
            "Facebook": ["FB"],
            "Meta": ["Meta"],
            "Google": ["GOOGL"],
            "Salesforce": ["SF.com"],
            "Oracle": ["Oracle-DB"]
        }
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
    
    result = guardrail.guard(
        test_case['input_text'],
        custom_terms=test_case['custom_terms']
    )
    expected = test_case['expected_entities']
    
    # Validate results
    matches = validate_entities(result.detected_entities, expected)
    
    print(f"Test Status: {'✓ PASS' if matches else '✗ FAIL'}")
    print(f"Contains Restricted Terms: {result.contains_entities}")
    
    if not matches:
        print("\nEntity Comparison:")
        all_entity_types = set(list(result.detected_entities.keys()) + list(expected.keys()))
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
