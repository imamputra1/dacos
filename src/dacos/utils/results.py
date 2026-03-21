# ruff: noqa: ARG002
"""
Industrial-grade error handling for dacos trading system.
Implements Ok and Err monads with comprehensive operations.
All type hints use Python 3.12+ syntax (| instead of Union, Optional).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    NoReturn,
    ParamSpec,
    TypeVar,
    assert_never,
)

# ====================== TYPE VARIABLES ======================

T = TypeVar("T")  # Success type
E = TypeVar("E")  # Error type (must be Exception or str)
U = TypeVar("U")  # Return type for transformations
F = TypeVar("F")  # New error type for error transformations
P = ParamSpec("P")  # Function parameters

# ====================== ALGEBRAIC RESULT TYPES ======================


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Success container - immutable monadic value."""

    value: T

    # ========== MONADIC CORE OPERATIONS ==========

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_err(self) -> NoReturn:
        raise ValueError("Cannot unwrap_err on Ok")

    def unwrap_or(self, default: T) -> T:
        return self.value

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        return self.value

    def expect(self, msg: str) -> T:
        return self.value

    def expect_err(self, msg: str) -> NoReturn:
        raise ValueError(f"{msg}: Expected Err, got Ok")

    # ========== TRANSFORMATION OPERATIONS ==========

    def map(self, op: Callable[[T], U]) -> Ok[U]:
        return Ok(op(self.value))

    def map_err(self, op: Callable[[E], F]) -> Ok[T]:
        return self

    def and_then(self, op: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return op(self.value)

    def or_else(self, op: Callable[[E], Result[T, F]]) -> Ok[T]:
        return self

    # ========== COMBINATION OPERATIONS ==========

    def and_(self, res: Result[U, E]) -> Result[U, E]:
        return res

    def or_(self, res: Result[T, F]) -> Ok[T]:
        return self

    # ========== INSPECTION OPERATIONS ==========

    def ok(self) -> T | None:
        return self.value

    def err(self) -> None:
        return None

    def contains(self, value: Any) -> bool:
        return self.value == value

    def contains_err(self, error: Any) -> bool:
        return False

    # ========== ITERATION ==========

    def __iter__(self):
        yield self.value

    def iter(self):
        yield self.value


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    """Error container - immutable monadic value."""

    error: E

    # ========== MONADIC CORE OPERATIONS ==========

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> NoReturn:
        raise ValueError(f"Cannot unwrap Err: {self.error}")

    def unwrap_err(self) -> E:
        return self.error

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        return f()

    def expect(self, msg: str) -> NoReturn:
        raise ValueError(f"{msg}: {self.error}")

    def expect_err(self, msg: str) -> E:
        return self.error

    # ========== TRANSFORMATION OPERATIONS ==========

    def map(self, op: Callable[[T], U]) -> Err[E]:
        return self

    def map_err(self, op: Callable[[E], F]) -> Err[F]:
        return Err(op(self.error))

    def and_then(self, op: Callable[[T], Result[U, E]]) -> Err[E]:
        return self

    def or_else(self, op: Callable[[E], Result[T, F]]) -> Result[T, F]:
        return op(self.error)

    # ========== COMBINATION OPERATIONS ==========

    def and_(self, res: Result[U, E]) -> Err[E]:
        return self

    def or_(self, res: Result[T, F]) -> Result[T, F]:
        return res

    # ========== INSPECTION OPERATIONS ==========

    def ok(self) -> None:
        return None

    def err(self) -> E | None:
        return self.error

    def contains(self, value: Any) -> bool:
        return False

    def contains_err(self, error: Any) -> bool:
        return self.error == error

    # ========== ITERATION ==========

    def __iter__(self):
        return iter(())

    def iter(self):
        return iter(())


# ====================== RESULT TYPE ALIAS ======================

Result = Ok[T] | Err[E]


def match_result(
    result: Result[T, E],
    on_ok: Callable[[T], U],
    on_err: Callable[[E], U],
) -> U:
    """Pattern matching with type safety."""
    if isinstance(result, Ok):
        return on_ok(result.value)
    elif isinstance(result, Err):
        return on_err(result.error)
    else:
        assert_never(result)


# ====================== DECORATORS FOR EXCEPTION HANDLING ======================


def safe(func: Callable[P, T]) -> Callable[P, Result[T, Exception]]:
    """Wrap synchronous function to return Result monad."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, Exception]:
        try:
            return Ok(func(*args, **kwargs))
        except Exception as e:
            logging.debug(f"Function {func.__name__} failed: {e}")
            return Err(e)

    return wrapper


def safe_async(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[Result[T, Exception]]]:
    """Wrap asynchronous function to return Result monad."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, Exception]:
        try:
            result = await func(*args, **kwargs)
            return Ok(result)
        except Exception as e:
            logging.debug(f"Async function {func.__name__} failed: {e}")
            return Err(e)

    return wrapper


# ====================== MONADIC COMPOSITION UTILITIES ======================


class ResultBuilder(Generic[E]):
    """Monadic builder for chaining Result operations."""

    @staticmethod
    def of(value: T) -> Result[T, E]:
        return Ok(value)

    @staticmethod
    def fail(error: E) -> Result[Any, E]:
        return Err(error)

    @staticmethod
    def collect(results: list[Result[T, E]]) -> Result[list[T], E]:
        """Collect multiple Results, fail fast on first error."""
        collected = []
        for result in results:
            if isinstance(result, Err):
                return result
            collected.append(result.value)
        return Ok(collected)

    @staticmethod
    def collect_all(results: list[Result[T, E]]) -> Result[list[T], list[E]]:
        """Collect all Results, gather all errors."""
        values = []
        errors = []
        for result in results:
            match_result(
                result,
                on_ok=lambda v: values.append(v),
                on_err=lambda e: errors.append(e),
            )
        if errors:
            return Err(errors)
        return Ok(values)

    @staticmethod
    def sequence(*results: Result[T, E]) -> Result[tuple[T, ...], E]:
        """Sequence multiple Results into a tuple."""
        values = []
        for result in results:
            if isinstance(result, Err):
                return result
            values.append(result.value)
        return Ok(tuple(values))


# ====================== ERROR RECOVERY PATTERNS ======================


class RetryConfig:
    """Configuration for retry operations."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        max_delay: float = 10.0,
        retry_on: type[Exception] | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retry_on = retry_on or Exception


def with_retry(
    config: RetryConfig | None = None,
) -> Callable[[Callable[P, Result[T, E]]], Callable[P, Result[T, E]]]:
    """Decorator to add retry logic to Result-returning functions."""
    cfg = config or RetryConfig()

    def decorator(func: Callable[P, Result[T, E]]) -> Callable[P, Result[T, E]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, E]:
            last_error: E | None = None
            for attempt in range(cfg.max_retries + 1):
                result = func(*args, **kwargs)
                if isinstance(result, Ok):
                    return result
                # Must be Err
                last_error = result.error
                if not isinstance(last_error, cfg.retry_on):
                    return result
                if attempt < cfg.max_retries:
                    delay = min(
                        cfg.backoff_factor**attempt,
                        cfg.max_delay,
                    )
                    time.sleep(delay)
            # After exhausting retries, last_error is guaranteed to be set
            assert last_error is not None
            return Err(last_error)

        return wrapper

    return decorator


async def with_retry_async(
    func: Callable[P, Awaitable[Result[T, E]]],
    config: RetryConfig | None = None,
) -> Callable[P, Awaitable[Result[T, E]]]:
    """Async version of with_retry."""
    cfg = config or RetryConfig()

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, E]:
        last_error: E | None = None
        for attempt in range(cfg.max_retries + 1):
            result = await func(*args, **kwargs)
            if isinstance(result, Ok):
                return result
            last_error = result.error
            if not isinstance(last_error, cfg.retry_on):
                return result
            if attempt < cfg.max_retries:
                delay = min(
                    cfg.backoff_factor**attempt,
                    cfg.max_delay,
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        return Err(last_error)

    return wrapper


# ====================== OPTION TYPE (FOR COMPLETENESS) ======================


@dataclass(frozen=True, slots=True)
class Some(Generic[T]):
    """Some value container (Rust's Some)."""

    value: T

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def map(self, f: Callable[[T], U]) -> Some[U]:
        return Some(f(self.value))

    def and_then(self, f: Callable[[T], Option[U]]) -> Option[U]:
        return f(self.value)


@dataclass(frozen=True, slots=True)
class _None:
    """None value container (Rust's None)."""

    def is_some(self) -> bool:
        return False

    def is_none(self) -> bool:
        return True

    def unwrap(self) -> NoReturn:
        raise ValueError("Cannot unwrap None")

    def unwrap_or(self, default: T) -> T:
        return default

    def map(self, f: Callable[[T], U]) -> _None:
        return self

    def and_then(self, f: Callable[[T], Option[U]]) -> _None:
        return self


Option = Some[T] | _None
NoneType = _None()


# ====================== ADVANCED MONADIC OPERATIONS ======================


def try_all(*operations: Callable[[], Result[T, E]]) -> Result[list[T], list[E]]:
    """Try all operations, collect all results/errors."""
    results = []
    errors = []
    for op in operations:
        result = op()
        match_result(
            result,
            on_ok=lambda v: results.append(v),
            on_err=lambda e: errors.append(e),
        )
    if errors:
        return Err(errors)
    return Ok(results)


def fallback(
    primary: Callable[[], Result[T, E]],
    *fallbacks: Callable[[], Result[T, E]],
) -> Result[T, list[E]]:
    """Try primary, then fallbacks sequentially."""
    errors = []
    result = primary()
    if isinstance(result, Ok):
        return result
    errors.append(result.error)
    for fallback_op in fallbacks:
        result = fallback_op()
        if isinstance(result, Ok):
            return result
        errors.append(result.error)
    return Err(errors)


# ====================== CONTEXT MANAGER SUPPORT ======================


@contextmanager
def result_context() -> Any:
    """Context manager for Result operations."""
    try:
        yield
    except Exception as e:
        return Err(e)


class ResultContext:
    """Context manager with automatic Result wrapping."""

    def __enter__(self) -> ResultContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val:
            return isinstance(exc_val, Exception)
        return True

    @staticmethod
    def wrap(func: Callable[P, T]) -> Callable[P, Result[T, Exception]]:
        """Wrap function in context manager."""

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, Exception]:
            with ResultContext():
                try:
                    return Ok(func(*args, **kwargs))
                except Exception as e:
                    return Err(e)

        return wrapper


# ====================== PERFORMANCE MONITORING ======================


class MonadMetrics:
    """Metrics collection for monadic operations."""

    def __init__(self) -> None:
        self.success_count = 0
        self.error_count = 0
        self.total_operations = 0

    def record(self, result: Result[Any, Any]) -> None:
        self.total_operations += 1
        if isinstance(result, Ok):
            self.success_count += 1
        else:
            self.error_count += 1

    @property
    def success_rate(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return self.success_count / self.total_operations


# ====================== TYPE GUARDS & VALIDATORS ======================


def is_ok(result: Result[Any, Any]) -> bool:
    return isinstance(result, Ok)


def is_err(result: Result[Any, Any]) -> bool:
    return isinstance(result, Err)


def as_optional(result: Result[T, E]) -> T | None:
    return result.ok()


def from_optional(value: T | None, error: E) -> Result[T, E]:
    if value is None:
        return Err(error)
    return Ok(value)


# ====================== EXPORTS ======================

__all__ = [
    # Core Monadic Types
    "Result",
    "Ok",
    "Err",
    # Option Types
    "Option",
    "Some",
    "NoneType",
    # Core Functions
    "match_result",
    "safe",
    "safe_async",
    # Builders
    "ResultBuilder",
    # Error Recovery
    "RetryConfig",
    "with_retry",
    "with_retry_async",
    # Advanced Operations
    "try_all",
    "fallback",
    # Context Managers
    "ResultContext",
    "result_context",
    # Performance
    "MonadMetrics",
    # Type Guards
    "is_ok",
    "is_err",
    "as_optional",
    "from_optional",
]
