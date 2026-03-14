"""
dacos.utils - Utility modules for dacos library.
"""

from dacos.utils.results import (
    Err,
    # Performance
    MonadMetrics,
    NoneType,
    Ok,
    # Option Types
    Option,
    # Core Monadic Types
    Result,
    # Builders
    ResultBuilder,
    # Context Managers
    ResultContext,
    # Error Recovery
    RetryConfig,
    Some,
    as_optional,
    fallback,
    from_optional,
    is_err,
    # Type Guards
    is_ok,
    # Core Functions
    match_result,
    result_context,
    safe,
    safe_async,
    # Advanced Operations
    try_all,
    with_retry,
    with_retry_async,
)

__all__ = [
    "Result",
    "Ok",
    "Err",
    "Option",
    "Some",
    "NoneType",
    "match_result",
    "safe",
    "safe_async",
    "ResultBuilder",
    "RetryConfig",
    "with_retry",
    "with_retry_async",
    "try_all",
    "fallback",
    "ResultContext",
    "result_context",
    "MonadMetrics",
    "is_ok",
    "is_err",
    "as_optional",
    "from_optional",
]
