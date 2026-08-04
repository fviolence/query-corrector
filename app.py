# SPDX-License-Identifier: 0BSD
"""Conservative LanguageTool adapter for the SearXNG query-corrector engine."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import unicodedata

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Annotated

import httpx

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field


LOGGER = logging.getLogger("query_corrector")
_MONOTONIC = time.monotonic

_ALLOWED_FORMAT_CHARACTERS = frozenset(("\u200c", "\u200d"))
_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_log_level(name: str, default: str) -> int:
    value = os.getenv(name, default).strip().upper()
    try:
        return _LOG_LEVELS[value]
    except KeyError as exc:
        choices = ", ".join(_LOG_LEVELS)
        raise ValueError(f"{name} must be one of: {choices}") from exc


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_language_variants(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _parse_csv(value):
        source, separator, target = item.partition(":")
        if not separator or not source.strip() or not target.strip():
            raise ValueError(
                "LANGUAGE_VARIANTS must contain comma-separated source:target mappings"
            )
        result[source.strip().casefold()] = target.strip()
    return result


@dataclass(slots=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    languagetool_url: str
    languagetool_timeout: float
    default_language: str
    preferred_variants: tuple[str, ...]
    language_variants: dict[str, str]
    ignored_words: frozenset[str]
    max_query_length: int
    max_correction_length: int
    max_languagetool_response_bytes: int
    max_edits: int
    max_token_edit_distance: int
    min_token_length: int
    allow_whitespace_replacements: bool
    log_level: int

    @classmethod
    def from_env(cls) -> "Settings":
        url = os.getenv("LANGUAGETOOL_URL", "http://languagetool:8010").rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("LANGUAGETOOL_URL must be an absolute HTTP(S) URL")

        return cls(
            languagetool_url=url,
            languagetool_timeout=_env_float(
                "LANGUAGETOOL_TIMEOUT", default=0.8, minimum=0.05, maximum=30.0
            ),
            default_language=os.getenv("DEFAULT_LANGUAGE", "auto").strip() or "auto",
            preferred_variants=_parse_csv(
                os.getenv("PREFERRED_VARIANTS", "en-US,de-DE,pt-PT")
            ),
            language_variants=_parse_language_variants(
                os.getenv("LANGUAGE_VARIANTS", "en:en-US,de:de-DE,pt:pt-PT")
            ),
            ignored_words=frozenset(
                word.casefold() for word in _parse_csv(os.getenv("IGNORED_WORDS", ""))
            ),
            max_query_length=_env_int(
                "MAX_QUERY_LENGTH", default=80, minimum=1, maximum=4096
            ),
            max_correction_length=_env_int(
                "MAX_CORRECTION_LENGTH", default=256, minimum=1, maximum=8192
            ),
            max_languagetool_response_bytes=_env_int(
                "MAX_LANGUAGETOOL_RESPONSE_BYTES",
                default=1_048_576,
                minimum=1_024,
                maximum=16_777_216,
            ),
            max_edits=_env_int("MAX_EDITS", default=2, minimum=1, maximum=20),
            max_token_edit_distance=_env_int(
                "MAX_TOKEN_EDIT_DISTANCE", default=2, minimum=1, maximum=10
            ),
            min_token_length=_env_int(
                "MIN_TOKEN_LENGTH", default=3, minimum=1, maximum=32
            ),
            allow_whitespace_replacements=_env_bool(
                "ALLOW_WHITESPACE_REPLACEMENTS", default=False
            ),
            log_level=_env_log_level("LOG_LEVEL", default="WARNING"),
        )


settings = Settings.from_env()


def _configure_logging() -> None:
    """Configure this service's logger without modifying Uvicorn's loggers."""

    LOGGER.setLevel(settings.log_level)
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        LOGGER.addHandler(handler)
    LOGGER.propagate = False


class CorrectionRequest(BaseModel):
    """Request accepted from the SearXNG query-corrector engine."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1)
    language: str | None = None


class CorrectionResponse(BaseModel):
    """Response consumed by the SearXNG query-corrector engine."""

    correction: str | None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    timeout = httpx.Timeout(settings.languagetool_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        app.state.languagetool_client = client
        app.state.supported_languages = None
        yield


app = FastAPI(
    title="SearXNG Query Corrector",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


async def get_languagetool_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.languagetool_client


LanguageToolClient = Annotated[httpx.AsyncClient, Depends(get_languagetool_client)]


def normalize_language(language: str | None) -> str:
    """Map SearXNG language values to a LanguageTool-compatible language."""

    normalized = (language or "").strip()
    if normalized.casefold() in {"", "all", "auto"}:
        normalized = settings.default_language
    return settings.language_variants.get(normalized.casefold(), normalized)


def _utf16_offset_to_index(text: str, offset: int) -> int | None:
    """Convert LanguageTool's UTF-16 code-unit offset to a Python string index."""

    if offset < 0:
        return None

    encoded = text.encode("utf-16-le")
    byte_offset = offset * 2
    if byte_offset > len(encoded):
        return None

    try:
        return len(encoded[:byte_offset].decode("utf-16-le"))
    except UnicodeDecodeError:
        return None


def _normalized_query(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _without_allowed_format_characters(value: str) -> str:
    return "".join(char for char in value if char not in _ALLOWED_FORMAT_CHARACTERS)


def _is_equivalent_correction(correction: str, original_query: str) -> bool:
    """Mirror SearXNG's asymmetric ZWNJ/ZWJ equivalence check."""

    correction_key = _normalized_query(correction)
    original_key = _normalized_query(original_query)
    if correction_key == original_key:
        return True

    if _without_allowed_format_characters(
        correction_key
    ) != _without_allowed_format_characters(original_key):
        return False

    correction_format_count = sum(
        char in _ALLOWED_FORMAT_CHARACTERS for char in correction_key
    )
    original_format_count = sum(
        char in _ALLOWED_FORMAT_CHARACTERS for char in original_key
    )
    return correction_format_count < original_format_count


def _has_forbidden_query_syntax_or_controls(value: str) -> bool:
    """Return whether SearXNG would reject this correction as unsafe."""

    return any(token[0] in "!:<" for token in value.split()) or any(
        (char.isspace() and char != " ")
        or (
            unicodedata.category(char).startswith("C")
            and char not in _ALLOWED_FORMAT_CHARACTERS
        )
        for char in value
    )


def _is_plain_word(value: str, minimum_length: int = 1) -> bool:
    """Return whether a value is safe to treat as one natural-language word."""

    if len(value) < minimum_length or not any(char.isalpha() for char in value):
        return False

    allowed_punctuation = {"'", "’", "-"}
    return all(
        char.isalpha()
        or char in allowed_punctuation
        or char in _ALLOWED_FORMAT_CHARACTERS
        for char in value
    )


def _match_source_case(source: str, replacement: str) -> str:
    """Preserve the source token's simple capitalization pattern."""

    if source.islower():
        return replacement.lower()
    if source.isupper():
        return replacement.upper()
    if source.istitle():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _damerau_levenshtein(source: str, target: str, limit: int) -> int:
    """Return restricted Damerau-Levenshtein distance, stopping above ``limit``."""

    source = unicodedata.normalize("NFC", source).casefold()
    target = unicodedata.normalize("NFC", target).casefold()

    if abs(len(source) - len(target)) > limit:
        return limit + 1

    previous_previous: list[int] | None = None
    previous = list(range(len(target) + 1))

    for source_index, source_char in enumerate(source, start=1):
        current = [source_index]
        row_minimum = source_index

        for target_index, target_char in enumerate(target, start=1):
            cost = 0 if source_char == target_char else 1
            value = min(
                current[target_index - 1] + 1,
                previous[target_index] + 1,
                previous[target_index - 1] + cost,
            )

            if (
                previous_previous is not None
                and source_index > 1
                and target_index > 1
                and source_char == target[target_index - 2]
                and source[source_index - 2] == target_char
            ):
                value = min(value, previous_previous[target_index - 2] + 1)

            current.append(value)
            row_minimum = min(row_minimum, value)

        if row_minimum > limit:
            return limit + 1

        previous_previous, previous = previous, current

    return previous[-1]


def _first_safe_replacement(source: str, replacements: Any) -> str | None:
    if not isinstance(replacements, list):
        return None

    for replacement_obj in replacements:
        if not isinstance(replacement_obj, dict):
            continue

        replacement = replacement_obj.get("value")
        if not isinstance(replacement, str) or not replacement:
            continue
        if replacement != replacement.strip():
            continue
        if any(
            unicodedata.category(char).startswith("C")
            and char not in _ALLOWED_FORMAT_CHARACTERS
            for char in replacement
        ):
            continue
        if any(char.isspace() and char != " " for char in replacement):
            continue
        if settings.allow_whitespace_replacements:
            words = replacement.split(" ")
            if any(not word or not _is_plain_word(word) for word in words):
                continue
        elif " " in replacement or not _is_plain_word(replacement):
            continue
        replacement = _match_source_case(source, replacement)
        if _is_equivalent_correction(replacement, source):
            continue
        if (
            _damerau_levenshtein(source, replacement, settings.max_token_edit_distance)
            > settings.max_token_edit_distance
        ):
            continue
        return replacement

    return None


def build_correction(query: str, payload: Any) -> str | None:
    """Apply conservative, non-overlapping spelling corrections."""

    if not isinstance(payload, dict):
        return None

    matches = payload.get("matches")
    if not isinstance(matches, list):
        return None

    edits_by_span: dict[tuple[int, int], str] = {}

    for match in matches:
        if not isinstance(match, dict):
            continue

        rule = match.get("rule")
        if not isinstance(rule, dict):
            continue
        if str(rule.get("issueType", "")).casefold() != "misspelling":
            continue

        offset = match.get("offset")
        length = match.get("length")
        if (
            isinstance(offset, bool)
            or isinstance(length, bool)
            or not isinstance(offset, int)
            or not isinstance(length, int)
            or length <= 0
        ):
            continue

        start = _utf16_offset_to_index(query, offset)
        end = _utf16_offset_to_index(query, offset + length)
        if start is None or end is None or not 0 <= start < end <= len(query):
            continue

        source = query[start:end]
        if source.casefold() in settings.ignored_words:
            continue
        if not _is_plain_word(source, settings.min_token_length):
            continue

        replacement = _first_safe_replacement(source, match.get("replacements"))
        if replacement is None:
            continue

        edits_by_span.setdefault((start, end), replacement)

    if not edits_by_span or len(edits_by_span) > settings.max_edits:
        return None

    edits = sorted(
        (start, end, replacement)
        for (start, end), replacement in edits_by_span.items()
    )
    for previous, current in zip(edits, edits[1:]):
        if current[0] < previous[1]:
            return None

    correction = query
    for start, end, replacement in reversed(edits):
        correction = correction[:start] + replacement + correction[end:]

    correction = correction.strip()
    if (
        not correction
        or len(correction) > settings.max_correction_length
        or _is_equivalent_correction(correction, query.strip())
        or _has_forbidden_query_syntax_or_controls(correction)
    ):
        return None

    return correction


async def _bounded_languagetool_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    timeout: float,
    data: dict[str, str] | None = None,
) -> httpx.Response:
    """Read a LanguageTool response without exceeding the configured body limit."""

    async with client.stream(
        method,
        f"{settings.languagetool_url}{path}",
        data=data,
        headers={"Accept": "application/json"},
        timeout=timeout,
    ) as response:
        content_length = response.headers.get("content-length")
        if content_length is not None:
            # Content-Length describes the bytes transferred on the wire and may
            # therefore refer to a compressed representation.  This is only an
            # early rejection; the streaming loop below enforces the limit on
            # the content-decoded body returned by ``aiter_bytes()``.
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = 0
            if declared_length > settings.max_languagetool_response_bytes:
                raise ValueError("LanguageTool response exceeds the configured size limit")

        body = bytearray()
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > settings.max_languagetool_response_bytes:
                raise ValueError("LanguageTool response exceeds the configured size limit")

        # ``aiter_bytes()`` returns content-decoded bytes.  Do not copy framing
        # or content-coding headers to the reconstructed response: doing so
        # would make httpx decode an already-decoded body a second time.
        passthrough_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower()
            not in {"content-encoding", "content-length", "transfer-encoding"}
        }
        return httpx.Response(
            response.status_code,
            headers=passthrough_headers,
            content=bytes(body),
            request=response.request,
        )


def _supported_language_codes(payload: Any) -> frozenset[str]:
    """Extract case-insensitive language codes from ``/v2/languages``."""

    if not isinstance(payload, list):
        return frozenset()

    codes: set[str] = set()
    for language in payload:
        if not isinstance(language, dict):
            continue
        for key in ("code", "longCode"):
            value = language.get(key)
            if isinstance(value, str) and value.strip():
                codes.add(value.strip().casefold())
    return frozenset(codes)


async def _fetch_supported_languages(
    client: httpx.AsyncClient,
) -> frozenset[str]:
    response = await _bounded_languagetool_request(
        client,
        "GET",
        "/v2/languages",
        timeout=settings.languagetool_timeout,
    )
    response.raise_for_status()
    languages = _supported_language_codes(response.json())
    if not languages:
        raise ValueError("LanguageTool returned no supported languages")
    return languages


def _languagetool_form(query: str, language: str) -> dict[str, str]:
    form_data = {"text": query, "language": language}
    if language.casefold() == "auto" and settings.preferred_variants:
        form_data["preferredVariants"] = ",".join(settings.preferred_variants)
    return form_data


async def _check_with_languagetool(
    client: httpx.AsyncClient,
    query: str,
    language: str,
) -> httpx.Response:
    # The outer asyncio timeout makes LANGUAGETOOL_TIMEOUT a hard wall-clock
    # budget across connection setup, response streaming, and the optional retry.
    async with asyncio.timeout(settings.languagetool_timeout):
        deadline = _MONOTONIC() + settings.languagetool_timeout
        response = await _bounded_languagetool_request(
            client,
            "POST",
            "/v2/check",
            data=_languagetool_form(query, language),
            timeout=settings.languagetool_timeout,
        )

        if response.status_code == status.HTTP_400_BAD_REQUEST and language.casefold() != "auto":
            remaining_timeout = deadline - _MONOTONIC()
            if remaining_timeout < 0.05:
                LOGGER.info(
                    "LanguageTool rejected language %s; no timeout budget remains for auto detection",
                    language,
                )
                return response

            LOGGER.info(
                "LanguageTool rejected language %s; retrying with auto detection",
                language,
            )
            response = await _bounded_languagetool_request(
                client,
                "POST",
                "/v2/check",
                data=_languagetool_form(query, "auto"),
                timeout=remaining_timeout,
            )

        return response


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readiness(request: Request, client: LanguageToolClient) -> dict[str, str]:
    try:
        async with asyncio.timeout(settings.languagetool_timeout):
            request.app.state.supported_languages = await _fetch_supported_languages(client)
    except (httpx.HTTPError, ValueError, TimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LanguageTool is unavailable",
        ) from exc

    return {"status": "ready"}


@app.post("/v1/correct", response_model=CorrectionResponse)
async def correct(
    request_data: CorrectionRequest,
    request: Request,
    client: LanguageToolClient,
) -> CorrectionResponse:
    query = request_data.query.strip()
    if (
        not query
        or len(query) > settings.max_query_length
        or _has_forbidden_query_syntax_or_controls(query)
    ):
        return CorrectionResponse(correction=None)

    language = normalize_language(request_data.language)
    supported_languages = getattr(request.app.state, "supported_languages", None)
    if (
        supported_languages is not None
        and language.casefold() != "auto"
        and language.casefold() not in supported_languages
    ):
        LOGGER.info(
            "LanguageTool does not support language %s; using auto detection",
            language,
        )
        language = "auto"

    try:
        response = await _check_with_languagetool(client, query, language)
        response.raise_for_status()
        payload = response.json()
    except (httpx.TimeoutException, TimeoutError) as exc:
        LOGGER.warning("LanguageTool request timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LanguageTool request timed out",
        ) from exc
    except (httpx.HTTPError, ValueError) as exc:
        LOGGER.warning("LanguageTool request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LanguageTool request failed",
        ) from exc

    return CorrectionResponse(correction=build_correction(query, payload))
