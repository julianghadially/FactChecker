"""Quick test script to verify JudgeModule enhancements.

This script tests the basic functionality without requiring LLM configuration.
It verifies:
1. Module initialization with/without web search
2. Lazy initialization of web services
3. Knowledge limitation detection
4. Query extraction
"""

from src.factchecker.simple.modules.judge_module import JudgeModule


def test_initialization():
    """Test module initialization."""
    print("Test 1: Module Initialization")
    print("-" * 50)

    # Test with web search enabled (default)
    judge1 = JudgeModule()
    assert judge1.use_web_search is True
    print("✓ Default initialization: web_search=True")

    # Test with web search disabled
    judge2 = JudgeModule(use_web_search=False)
    assert judge2.use_web_search is False
    print("✓ Custom initialization: web_search=False")

    print()


def test_lazy_initialization():
    """Test lazy initialization of web services."""
    print("Test 2: Lazy Initialization of Web Services")
    print("-" * 50)

    judge = JudgeModule(use_web_search=True)

    # Services should not be initialized yet
    assert judge._serper_service is None
    assert judge._firecrawl_service is None
    print("✓ Services not initialized on module creation")

    # Access properties to trigger initialization
    serper = judge.serper_service
    firecrawl = judge.firecrawl_service

    assert judge._serper_service is not None
    assert judge._firecrawl_service is not None
    print("✓ Services initialized on first access")

    print()


def test_knowledge_limitation_detection():
    """Test knowledge limitation detection."""
    print("Test 3: Knowledge Limitation Detection")
    print("-" * 50)

    judge = JudgeModule()

    # Test cases that should trigger detection
    trigger_cases = [
        "I cannot verify this due to my knowledge cutoff.",
        "After my training data, I do not have information.",
        "This is a recent event I cannot confirm.",
        "I don't have access to information beyond my training data.",
    ]

    for reasoning in trigger_cases:
        result = judge._detect_knowledge_limitation(reasoning)
        assert result is True, f"Failed to detect: {reasoning}"
    print(f"✓ Correctly detected {len(trigger_cases)} cases with limitations")

    # Test cases that should NOT trigger detection
    non_trigger_cases = [
        "The Earth orbits the Sun, which is a well-established fact.",
        "Water freezes at 0°C at standard pressure.",
        "This statement is supported by scientific consensus.",
    ]

    for reasoning in non_trigger_cases:
        result = judge._detect_knowledge_limitation(reasoning)
        assert result is False, f"False positive: {reasoning}"
    print(f"✓ Correctly rejected {len(non_trigger_cases)} cases without limitations")

    print()


def test_query_extraction():
    """Test search query extraction."""
    print("Test 4: Search Query Extraction")
    print("-" * 50)

    judge = JudgeModule()

    statements = [
        "SpaceX launched Starship Flight 6 in November 2024.",
        "The 2024 US Presidential election results were announced in November.",
        "Paris hosted the 2024 Summer Olympics.",
    ]

    for statement in statements:
        query = judge._extract_search_query(statement)
        assert query == statement  # Current implementation returns statement as-is
        print(f"✓ Extracted query for: {statement[:50]}...")

    print()


def test_module_structure():
    """Test module structure and attributes."""
    print("Test 5: Module Structure")
    print("-" * 50)

    judge = JudgeModule()

    # Check key attributes exist
    assert hasattr(judge, 'judge')
    assert hasattr(judge, 'use_web_search')
    assert hasattr(judge, 'UNCERTAINTY_KEYWORDS')
    print("✓ Module has required attributes")

    # Check methods exist
    assert hasattr(judge, '_detect_knowledge_limitation')
    assert hasattr(judge, '_extract_search_query')
    assert hasattr(judge, '_gather_web_evidence')
    assert hasattr(judge, 'forward')
    print("✓ Module has required methods")

    # Check uncertainty keywords
    assert len(judge.UNCERTAINTY_KEYWORDS) > 10
    print(f"✓ Module has {len(judge.UNCERTAINTY_KEYWORDS)} uncertainty keywords")

    print()


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("JudgeModule Enhancement Tests")
    print("=" * 50)
    print()

    try:
        test_initialization()
        test_lazy_initialization()
        test_knowledge_limitation_detection()
        test_query_extraction()
        test_module_structure()

        print("=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
