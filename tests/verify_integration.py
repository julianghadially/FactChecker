"""Verification script to ensure HybridJudgeModule integration is complete."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("HYBRID JUDGE MODULE - INTEGRATION VERIFICATION")
print("=" * 80)

# Test 1: Import verification
print("\n[1/5] Verifying imports...")
try:
    from src.factchecker.modules import HybridJudgeModule
    from src.factchecker.signatures import TemporalDetector
    from src.factchecker.simple.modules import JudgeModule
    from src.factchecker.modules import FactCheckerPipeline
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Class instantiation
print("\n[2/5] Verifying class instantiation...")
try:
    import dspy
    from src.context_.context import openai_key

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=openai_key))

    hybrid = HybridJudgeModule()
    print("✅ HybridJudgeModule instantiated successfully")
    print(f"   - temporal_detector: {type(hybrid.temporal_detector).__name__}")
    print(f"   - simple_judge: {type(hybrid.simple_judge).__name__}")
    print(f"   - fact_checker: {type(hybrid.fact_checker).__name__}")
except Exception as e:
    print(f"❌ Instantiation failed: {e}")
    sys.exit(1)

# Test 3: Module structure verification
print("\n[3/5] Verifying module structure...")
try:
    assert hasattr(hybrid, 'temporal_detector'), "Missing temporal_detector"
    assert hasattr(hybrid, 'simple_judge'), "Missing simple_judge"
    assert hasattr(hybrid, 'fact_checker'), "Missing fact_checker"
    assert hasattr(hybrid, 'forward'), "Missing forward method"
    print("✅ Module structure verified")
except AssertionError as e:
    print(f"❌ Structure verification failed: {e}")
    sys.exit(1)

# Test 4: TemporalDetector functionality
print("\n[4/5] Verifying TemporalDetector...")
try:
    detector = dspy.Predict(TemporalDetector)

    # Test temporal claim
    result1 = detector(statement="In December 2025, Apple announced a $150B buyback.")
    assert hasattr(result1, 'requires_web_search'), "Missing requires_web_search field"
    assert hasattr(result1, 'reasoning'), "Missing reasoning field"
    print("✅ TemporalDetector working")
    print(f"   - Test claim: 'In December 2025...'")
    print(f"   - requires_web_search: {result1.requires_web_search}")
    print(f"   - reasoning: {result1.reasoning[:80]}...")

except Exception as e:
    print(f"❌ TemporalDetector test failed: {e}")
    sys.exit(1)

# Test 5: Integration with gepa_optimize.py
print("\n[5/5] Verifying gepa_optimize.py integration...")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gepa_optimize",
        "src/optimizer/gepa_optimize.py"
    )
    gepa_module = importlib.util.module_from_spec(spec)

    # Check if file contains HybridJudgeModule import
    with open("src/optimizer/gepa_optimize.py", "r") as f:
        content = f.read()

    assert "from src.factchecker.modules.hybrid_judge_module import HybridJudgeModule" in content, \
        "Missing HybridJudgeModule import"
    assert "program = HybridJudgeModule()" in content, \
        "Not using HybridJudgeModule in program initialization"

    print("✅ gepa_optimize.py correctly uses HybridJudgeModule")
    print("   - Import statement: Found")
    print("   - Program initialization: Found")

except Exception as e:
    print(f"❌ gepa_optimize.py verification failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE - ALL CHECKS PASSED ✅")
print("=" * 80)
print("\nIntegration Status:")
print("  ✅ Imports working")
print("  ✅ Classes instantiable")
print("  ✅ Module structure correct")
print("  ✅ TemporalDetector functional")
print("  ✅ GEPA optimizer integration complete")
print("\nNext steps:")
print("  1. Run full tests: python tests/test_hybrid_judge.py")
print("  2. Run optimization: python -m src.optimizer.gepa_optimize --mlflow --auto light")
print("=" * 80)
