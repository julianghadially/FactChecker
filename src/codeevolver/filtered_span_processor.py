"""Drop noisy DSPy internal spans before they reach CodeEvolver's exporter.

DSPy + OpenInference emits a deep span tree for every LM call:

    pipeline -> JudgeModule.forward -> ChainOfThought.forward
        -> Predict.forward -> Predict(Signature).forward
        -> ChatAdapter.__call__ -> LM.__call__

Most of those intermediate nodes are structural plumbing. Only `LM.__call__`,
any `TOOL`/`RETRIEVER` spans, and the user's own `dspy.Module.forward` spans
carry signal a reflection LM can reason about.

`install_filter()` wraps every currently-registered SpanProcessor on the global
TracerProvider with `FilteredSpanProcessor`, which drops spans failing
`should_keep` before handing them to the underlying processor.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider


KEEP_KINDS = frozenset({"LLM", "TOOL", "RETRIEVER", "RERANKER", "EMBEDDING", "AGENT"})

DROP_NAMES = frozenset({
    "ChainOfThought.forward",
    "Predict.forward",
    "ChatAdapter.__call__",
    "ChatAdapter.format",
    "ChatAdapter.parse",
})

DROP_NAME_PREFIXES = ("Predict(",)


def should_keep(span: ReadableSpan) -> bool:
    kind = (span.attributes or {}).get("openinference.span.kind")
    if kind in KEEP_KINDS:
        return True
    name = span.name or ""
    if name in DROP_NAMES or name.startswith(DROP_NAME_PREFIXES):
        return False
    return True


class FilteredSpanProcessor(SpanProcessor):
    def __init__(self, wrapped: SpanProcessor):
        self._wrapped = wrapped

    def on_start(self, span, parent_context=None):
        self._wrapped.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        if should_keep(span):
            self._wrapped.on_end(span)

    def shutdown(self) -> None:
        self._wrapped.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._wrapped.force_flush(timeout_millis)


_installed = False


def install_filter() -> bool:
    """Wrap every processor on the global TracerProvider with the filter.

    Idempotent. Returns True if the filter is now active, False if the global
    provider isn't an SDK TracerProvider we can patch.
    """
    global _installed
    if _installed:
        return True
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        return False
    multi = getattr(provider, "_active_span_processor", None)
    processors = getattr(multi, "_span_processors", None)
    if processors is None:
        return False
    multi._span_processors = tuple(
        p if isinstance(p, FilteredSpanProcessor) else FilteredSpanProcessor(p)
        for p in processors
    )
    _installed = True
    return True
