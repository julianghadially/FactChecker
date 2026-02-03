# Temporal Awareness Module - Quick Start Guide

## 🚀 What Was Added

A preprocessing module that detects temporal signals (dates, years, "recently", etc.) and instructs the fact-checker to perform web searches for claims beyond the June 2024 knowledge cutoff.

---

## 📁 Files to Review

### New Files
1. **`src/factchecker/modules/temporal_awareness_module.py`** - Core implementation
2. **`example_temporal_awareness.py`** - Demo script
3. **`TEMPORAL_AWARENESS_README.md`** - Full documentation
4. **`IMPLEMENTATION_SUMMARY.md`** - Technical summary

### Modified Files
1. **`src/factchecker/models/data_types.py`** - Added `TemporalContext`
2. **`src/factchecker/modules/fact_checker_pipeline.py`** - Integrated temporal awareness
3. **`src/factchecker/modules/fire_judge_module.py`** - Accepts temporal context

---

## ⚡ How It Works (30 Second Version)

```python
# BEFORE: Statement about 2025
"Trump was inaugurated on January 20, 2025"
  ↓
[Claim Extraction]
  ↓
[Fire Judge] → "CONTAINS_UNSUPPORTED_CLAIMS" ❌
# (No web search, beyond knowledge cutoff)

# AFTER: Statement about 2025
"Trump was inaugurated on January 20, 2025"
  ↓
[Temporal Awareness] → Detects "2025", "January 20, 2025"
                     → Beyond cutoff = TRUE
                     → Generates search instructions
  ↓
[Claim Extraction]
  ↓
[Fire Judge] → Receives temporal context
            → "⚠️ TEMPORAL AWARENESS: Beyond cutoff..."
            → "ACTION REQUIRED: Perform web searches"
            → "Suggested: Add year filter 2025"
            → Performs web searches ✅
            → Returns accurate verdict ✅
```

---

## 🎯 Test It Now

```bash
# Run the demo script
python example_temporal_awareness.py
```

This will show:
- ✅ Temporal signal detection for various statements
- ✅ Context generation for post-cutoff dates
- ✅ Search strategy suggestions
- ✅ Integration with the pipeline

---

## 💡 Key Features

### 1. Automatic Detection
Finds temporal references:
- Explicit dates: "January 20, 2025"
- Years: "2025", "2024"
- Relative phrases: "recently", "this year", "last month"

### 2. Knowledge Cutoff Awareness
- Default cutoff: **June 2024**
- Flags anything beyond this date
- Configurable for future model updates

### 3. Search Strategy Suggestions
When beyond cutoff detected:
- 🔍 "Add year filter: 2025"
- 📰 "Use SerperService.search_news()"
- ⏰ "Apply recency filter: 'd', 'w', 'm'"

### 4. Explicit Judge Instructions
Context message example:
```
⚠️ TEMPORAL AWARENESS: This claim contains references to
events beyond the knowledge cutoff (June 2024).

🌐 ACTION REQUIRED: You MUST perform web searches to
verify this claim. Do not rely solely on pre-existing
knowledge.
```

---

## 🔧 Integration (Zero Code Changes Needed!)

If you're already using `FactCheckerPipeline`, it's automatically integrated:

```python
from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline

# Just use it normally - temporal awareness is automatic!
pipeline = FactCheckerPipeline()
result = pipeline(statement="Any statement about 2025...")

# The pipeline now:
# 1. Analyzes temporal signals
# 2. Extracts claims
# 3. Passes temporal context to judge
# 4. Performs temporal-aware searches
# 5. Returns accurate verdict
```

---

## 📊 Example Output

### Input
```python
statement = "The 2025 presidential inauguration was on January 20."
```

### Temporal Analysis
```python
TemporalContext(
    has_temporal_signals=True,
    is_beyond_cutoff=True,  # ← Triggers web search!
    temporal_entities=['2025', 'January 20'],
    suggested_search_modifiers=[
        'Add year filter: 2025',
        'Use news search for recent events'
    ],
    context_message="⚠️ TEMPORAL AWARENESS: This claim..."
)
```

### Result
Instead of "CONTAINS_UNSUPPORTED_CLAIMS", the system:
1. Searches web with year filter "2025"
2. Looks for news articles about inauguration
3. Returns accurate verdict based on evidence

---

## 🎓 Want to Learn More?

- **Full Documentation**: `TEMPORAL_AWARENESS_README.md`
- **Technical Details**: `IMPLEMENTATION_SUMMARY.md`
- **Code Examples**: `example_temporal_awareness.py`

---

## 🔥 Bottom Line

**Problem Solved**: Claims about 2025 or recent 2024 events are no longer automatically marked as "unsupported". The system now actively searches the web with temporal-aware queries.

**No Breaking Changes**: Existing code works exactly as before, now with improved accuracy for time-sensitive claims.

**Easy to Test**: Run `python example_temporal_awareness.py` to see it in action.

---

## 📞 Questions?

Check these in order:
1. `QUICK_START_GUIDE.md` ← You are here
2. `TEMPORAL_AWARENESS_README.md` ← Full documentation
3. `IMPLEMENTATION_SUMMARY.md` ← Technical details
4. `example_temporal_awareness.py` ← Code examples
