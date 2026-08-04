# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import asyncio
import gzip
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import pytest

from fastapi.testclient import TestClient

import app as shim


def spelling_match(query, source, replacement, *, issue_type="misspelling", offset=None):
    start = query.index(source) if offset is None else offset
    utf16_offset = len(query[:start].encode("utf-16-le")) // 2
    utf16_length = len(source.encode("utf-16-le")) // 2
    return {
        "offset": utf16_offset,
        "length": utf16_length,
        "replacements": [{"value": replacement}],
        "rule": {"issueType": issue_type},
    }


@pytest.fixture(autouse=True)
def restore_settings():
    original = {
        field: getattr(shim.settings, field)
        for field in shim.settings.__dataclass_fields__
    }
    yield
    for field, value in original.items():
        setattr(shim.settings, field, value)
    shim.app.dependency_overrides.clear()


def install_fake_client(fake):
    async def override_client():
        return fake

    shim.app.dependency_overrides[shim.get_languagetool_client] = override_client


def test_build_correction_applies_spelling_match():
    query = "search querry"
    payload = {"matches": [spelling_match(query, "querry", "query")]}
    assert shim.build_correction(query, payload) == "search query"


@pytest.mark.parametrize(
    ("source", "replacement", "expected"),
    [
        ("sysrem", "System", "system"),
        ("SYSREM", "System", "SYSTEM"),
        ("Sysrem", "system", "System"),
        ("iPone", "iPhone", "iPhone"),
    ],
)
def test_match_source_case(source, replacement, expected):
    assert shim._match_source_case(source, replacement) == expected


def test_build_correction_preserves_source_case():
    query = "sysrem shock reamke"
    payload = {
        "matches": [
            spelling_match(query, "sysrem", "System"),
            spelling_match(query, "reamke", "remake"),
        ]
    }
    assert shim.build_correction(query, payload) == "system shock remake"


def test_build_correction_ignores_non_spelling_match():
    query = "search querry"
    payload = {
        "matches": [
            spelling_match(query, "querry", "query", issue_type="grammar")
        ]
    }
    assert shim.build_correction(query, payload) is None


def test_build_correction_ignores_case_only_replacement():
    query = "arch linux"
    payload = {"matches": [spelling_match(query, "arch", "Arch")]}
    assert shim.build_correction(query, payload) is None


def test_build_correction_ignores_word_with_digits():
    query = "qwen3 model"
    payload = {"matches": [spelling_match(query, "qwen3", "qwen")]}
    assert shim.build_correction(query, payload) is None


def test_build_correction_honors_minimum_token_length():
    shim.settings.min_token_length = 4
    query = "teh query"
    payload = {"matches": [spelling_match(query, "teh", "the")]}
    assert shim.build_correction(query, payload) is None


def test_build_correction_ignores_large_edit_distance():
    query = "searxng settings"
    payload = {"matches": [spelling_match(query, "searxng", "searching")]}
    assert shim.build_correction(query, payload) is None


def test_build_correction_ignores_configured_word():
    shim.settings.ignored_words = frozenset({"archlinux"})
    query = "archlinux install"
    payload = {"matches": [spelling_match(query, "archlinux", "archlinix")]}
    assert shim.build_correction(query, payload) is None


def test_build_correction_rejects_whitespace_replacement_by_default():
    query = "someword"
    payload = {"matches": [spelling_match(query, query, "some word")]}
    assert shim.build_correction(query, payload) is None


def test_build_correction_allows_short_parts_in_multiword_replacement_when_enabled():
    shim.settings.allow_whitespace_replacements = True
    query = "alot"
    payload = {"matches": [spelling_match(query, query, "a lot")]}
    assert shim.build_correction(query, payload) == "a lot"


@pytest.mark.parametrize(
    "replacement",
    [
        "query1",
        "query?",
        "query  word",
        "query\tword",
    ],
)
def test_whitespace_mode_still_rejects_non_plain_replacements(replacement):
    shim.settings.allow_whitespace_replacements = True
    query = "query"
    payload = {"matches": [spelling_match(query, query, replacement)]}
    assert shim.build_correction(query, payload) is None


def test_build_correction_rejects_too_many_edits():
    query = "querry corection eror"
    payload = {
        "matches": [
            spelling_match(query, "querry", "query"),
            spelling_match(query, "corection", "correction"),
            spelling_match(query, "eror", "error"),
        ]
    }
    assert shim.build_correction(query, payload) is None


def test_build_correction_deduplicates_identical_spans():
    query = "search querry"
    payload = {
        "matches": [
            spelling_match(query, "querry", "query"),
            spelling_match(query, "querry", "quarry"),
        ]
    }
    assert shim.build_correction(query, payload) == "search query"


def test_build_correction_rejects_overlapping_matches():
    query = "querry"
    payload = {
        "matches": [
            spelling_match(query, "querry", "query"),
            {
                "offset": 1,
                "length": 3,
                "replacements": [{"value": "user"}],
                "rule": {"issueType": "misspelling"},
            },
        ]
    }
    assert shim.build_correction(query, payload) is None


def test_build_correction_handles_utf16_offsets():
    query = "😀 search querry"
    payload = {"matches": [spelling_match(query, "querry", "query")]}
    assert shim.build_correction(query, payload) == "😀 search query"


def test_utf16_offset_rejects_offset_past_end():
    assert shim._utf16_offset_to_index("test", 5) is None


def test_utf16_offset_rejects_surrogate_split():
    assert shim._utf16_offset_to_index("😀test", 1) is None


@pytest.mark.parametrize(
    "query",
    [
        "!google search querry",
        "search querry :de",
        "<1 search querry",
        "search querry !!ddg",
        "search\tquerry",
        "search\nquerry",
        "search \u202equerry",
        "search \u200bquerry",
    ],
)
def test_build_correction_rejects_results_searxng_would_consider_unsafe(query):
    payload = {"matches": [spelling_match(query, "querry", "query")]}
    assert shim.build_correction(query, payload) is None


def test_build_correction_accepts_required_joining_character_with_real_edit():
    query = "میرود"
    payload = {"matches": [spelling_match(query, query, "می\u200cروم")]}
    assert shim.build_correction(query, payload) == "می\u200cروم"


def test_build_correction_accepts_joining_character_insertion():
    query = "میرود"
    payload = {"matches": [spelling_match(query, query, "می\u200cرود")]}
    assert shim.build_correction(query, payload) == "می\u200cرود"


def test_build_correction_rejects_joining_character_deletion_only():
    query = "می\u200cرود"
    payload = {"matches": [spelling_match(query, query, "میرود")]}
    assert shim.build_correction(query, payload) is None


def test_damerau_levenshtein_counts_adjacent_transposition_once():
    assert shim._damerau_levenshtein("reamke", "remake", 2) == 1


def test_build_correction_honors_maximum_correction_length():
    shim.settings.max_correction_length = 11
    query = "search querry"
    payload = {"matches": [spelling_match(query, "querry", "query")]}
    assert shim.build_correction(query, payload) is None


def test_normalize_language_maps_variantless_language():
    assert shim.normalize_language("en") == "en-US"
    assert shim.normalize_language("de") == "de-DE"


def test_normalize_language_uses_default_for_missing_language():
    shim.settings.default_language = "auto"
    assert shim.normalize_language(None) == "auto"
    assert shim.normalize_language("all") == "auto"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MAX_EDITS", "x", "MAX_EDITS must be an integer"),
        ("MAX_EDITS", "0", "MAX_EDITS must be between 1 and 20"),
        (
            "MAX_LANGUAGETOOL_RESPONSE_BYTES",
            "512",
            "MAX_LANGUAGETOOL_RESPONSE_BYTES must be between 1024 and 16777216",
        ),
        (
            "LANGUAGETOOL_TIMEOUT",
            "x",
            "LANGUAGETOOL_TIMEOUT must be a number",
        ),
        (
            "LANGUAGETOOL_TIMEOUT",
            "31",
            "LANGUAGETOOL_TIMEOUT must be between 0.05 and 30.0",
        ),
        ("LOG_LEVEL", "verbose", "LOG_LEVEL must be one of"),
    ],
)
def test_settings_from_env_reports_invalid_values(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        shim.Settings.from_env()


def test_settings_from_env_rejects_invalid_language_variant(monkeypatch):
    monkeypatch.setenv("LANGUAGE_VARIANTS", "en-US")
    with pytest.raises(ValueError, match="LANGUAGE_VARIANTS"):
        shim.Settings.from_env()


@dataclass(frozen=True)
class StreamedBody:
    chunks: tuple[bytes, ...]


class ChunkedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class FakeLanguageToolClient:
    def __init__(
        self,
        response_payload=None,
        *,
        exception=None,
        post_results=None,
        get_result=None,
        get_exception=None,
        post_delay=0.0,
    ):
        self.response_payload = response_payload or {"matches": []}
        self.exception = exception
        self.post_results = list(post_results or [])
        self.get_result = get_result or (
            200,
            [{"code": "en", "longCode": "en-US"}],
        )
        self.get_exception = get_exception
        self.post_delay = post_delay
        self.posts = []
        self.gets = []

    @staticmethod
    def _response(method, url, result):
        status_code, payload, *metadata = result
        headers = metadata[0] if metadata else None
        request = httpx.Request(method, url)
        if isinstance(payload, StreamedBody):
            return httpx.Response(
                status_code,
                headers=headers,
                stream=ChunkedAsyncStream(payload.chunks),
                request=request,
            )
        if isinstance(payload, bytes):
            return httpx.Response(
                status_code,
                headers=headers,
                content=payload,
                request=request,
            )
        return httpx.Response(
            status_code,
            headers=headers,
            json=payload,
            request=request,
        )

    @asynccontextmanager
    async def stream(self, method, url, **kwargs):
        method = method.upper()
        if method == "POST":
            self.posts.append((url, kwargs))
            if self.post_delay:
                await asyncio.sleep(self.post_delay)
            if self.exception is not None:
                raise self.exception
            result = (
                self.post_results.pop(0)
                if self.post_results
                else (200, self.response_payload)
            )
        elif method == "GET":
            self.gets.append((url, kwargs))
            if self.get_exception is not None:
                raise self.get_exception
            result = self.get_result
        else:
            raise AssertionError(f"unexpected method: {method}")

        yield self._response(method, url, result)


def test_health_endpoint():
    with TestClient(shim.app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_endpoint_caches_supported_languages():
    fake = FakeLanguageToolClient(
        get_result=(
            200,
            [
                {"code": "en", "longCode": "en-US"},
                {"code": "de", "longCode": "de-DE"},
            ],
        )
    )
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.get("/readyz")
        supported_languages = client.app.state.supported_languages

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert supported_languages == frozenset({"en", "en-us", "de", "de-de"})


@pytest.mark.parametrize(
    "get_result",
    [
        (500, {"error": "unavailable"}),
        (200, b"not-json"),
        (200, []),
    ],
)
def test_readiness_endpoint_reports_unavailable_languagetool(get_result):
    fake = FakeLanguageToolClient(get_result=get_result)
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {"detail": "LanguageTool is unavailable"}


def test_endpoint_calls_languagetool_and_returns_correction():
    query = "search querry"
    fake = FakeLanguageToolClient(
        {"matches": [spelling_match(query, "querry", "query")]}
    )
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.post(
            "/v1/correct",
            json={"query": query, "language": "en"},
        )

    assert response.status_code == 200
    assert response.json() == {"correction": "search query"}

    url, kwargs = fake.posts[-1]
    assert url == "http://languagetool:8010/v2/check"
    assert kwargs["data"] == {"text": query, "language": "en-US"}
    assert kwargs["headers"] == {"Accept": "application/json"}


def test_endpoint_sends_preferred_variants_for_auto_language():
    fake = FakeLanguageToolClient()
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.post("/v1/correct", json={"query": "test query"})

    assert response.status_code == 200
    _, kwargs = fake.posts[-1]
    assert kwargs["data"]["language"] == "auto"
    assert kwargs["data"]["preferredVariants"] == "en-US,de-DE,pt-PT"


def test_endpoint_uses_auto_for_cached_unsupported_language():
    fake = FakeLanguageToolClient(
        get_result=(200, [{"code": "de", "longCode": "de-DE"}])
    )
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        assert client.get("/readyz").status_code == 200
        response = client.post(
            "/v1/correct",
            json={"query": "test query", "language": "en-US"},
        )

    assert response.status_code == 200
    _, kwargs = fake.posts[-1]
    assert kwargs["data"]["language"] == "auto"
    assert "preferredVariants" in kwargs["data"]


def test_endpoint_retries_rejected_language_with_auto_detection(monkeypatch):
    fake = FakeLanguageToolClient(
        post_results=[
            (400, {"error": "unsupported language"}),
            (200, {"matches": []}),
        ]
    )
    install_fake_client(fake)
    shim.settings.languagetool_timeout = 0.8
    monotonic_values = iter((100.0, 100.3))
    monkeypatch.setattr(shim, "_MONOTONIC", lambda: next(monotonic_values))

    with TestClient(shim.app) as client:
        response = client.post(
            "/v1/correct",
            json={"query": "test query", "language": "zz-ZZ"},
        )

    assert response.status_code == 200
    assert len(fake.posts) == 2
    assert fake.posts[0][1]["data"]["language"] == "zz-ZZ"
    assert fake.posts[0][1]["timeout"] == pytest.approx(0.8)
    assert fake.posts[1][1]["data"]["language"] == "auto"
    assert fake.posts[1][1]["timeout"] == pytest.approx(0.5)


def test_endpoint_skips_language_retry_when_timeout_budget_is_exhausted(monkeypatch):
    fake = FakeLanguageToolClient(
        post_results=[(400, {"error": "unsupported language"})]
    )
    install_fake_client(fake)
    shim.settings.languagetool_timeout = 0.8
    monotonic_values = iter((100.0, 100.76))
    monkeypatch.setattr(shim, "_MONOTONIC", lambda: next(monotonic_values))

    with TestClient(shim.app) as client:
        response = client.post(
            "/v1/correct",
            json={"query": "test query", "language": "zz-ZZ"},
        )

    assert response.status_code == 502
    assert len(fake.posts) == 1


def test_endpoint_tolerates_missing_supported_language_cache():
    fake = FakeLanguageToolClient()
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        del client.app.state.supported_languages
        response = client.post(
            "/v1/correct",
            json={"query": "test query", "language": "en-US"},
        )

    assert response.status_code == 200
    assert fake.posts[-1][1]["data"]["language"] == "en-US"


def test_endpoint_ignores_unexpected_request_field():
    fake = FakeLanguageToolClient()
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.post(
            "/v1/correct",
            json={"query": "test query", "unexpected": True},
        )

    assert response.status_code == 200
    assert len(fake.posts) == 1


def test_endpoint_fails_closed_for_oversized_query():
    fake = FakeLanguageToolClient()
    install_fake_client(fake)
    shim.settings.max_query_length = 5

    with TestClient(shim.app) as client:
        response = client.post("/v1/correct", json={"query": "too long"})

    assert response.status_code == 200
    assert response.json() == {"correction": None}
    assert fake.posts == []


@pytest.mark.parametrize(
    "query",
    [
        "!google search querry",
        "search querry :de",
        "search\nquerry",
        "search \u200bquerry",
    ],
)
def test_endpoint_skips_languagetool_for_searxng_unsafe_query(query):
    fake = FakeLanguageToolClient()
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.post("/v1/correct", json={"query": query})

    assert response.status_code == 200
    assert response.json() == {"correction": None}
    assert fake.posts == []


def test_endpoint_enforces_total_languagetool_timeout():
    fake = FakeLanguageToolClient(post_delay=0.05)
    install_fake_client(fake)
    shim.settings.languagetool_timeout = 0.01

    with TestClient(shim.app) as client:
        response = client.post("/v1/correct", json={"query": "test query"})

    assert response.status_code == 504


def test_endpoint_rejects_declared_oversized_languagetool_response():
    fake = FakeLanguageToolClient(post_results=[(200, b"x" * 11)])
    install_fake_client(fake)
    shim.settings.max_languagetool_response_bytes = 10

    with TestClient(shim.app) as client:
        response = client.post("/v1/correct", json={"query": "test query"})

    assert response.status_code == 502
    assert response.json() == {"detail": "LanguageTool request failed"}


def test_endpoint_rejects_streamed_oversized_languagetool_response():
    fake = FakeLanguageToolClient(
        post_results=[(200, StreamedBody((b"x" * 6, b"x" * 5)))]
    )
    install_fake_client(fake)
    shim.settings.max_languagetool_response_bytes = 10

    with TestClient(shim.app) as client:
        response = client.post("/v1/correct", json={"query": "test query"})

    assert response.status_code == 502
    assert response.json() == {"detail": "LanguageTool request failed"}


def test_endpoint_accepts_content_decoded_compressed_response():
    query = "search querry"
    payload = json.dumps(
        {"matches": [spelling_match(query, "querry", "query")]}
    ).encode()
    compressed_payload = gzip.compress(payload)
    fake = FakeLanguageToolClient(
        post_results=[
            (
                200,
                compressed_payload,
                {
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(compressed_payload)),
                    "Content-Type": "application/json",
                    "Transfer-Encoding": "chunked",
                },
            )
        ]
    )
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.post(
            "/v1/correct",
            json={"query": query, "language": "en-US"},
        )

    assert response.status_code == 200
    assert response.json() == {"correction": "search query"}


def test_endpoint_reports_upstream_timeout():
    request = httpx.Request("POST", "http://languagetool:8010/v2/check")
    fake = FakeLanguageToolClient(
        exception=httpx.ReadTimeout("timed out", request=request)
    )
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.post("/v1/correct", json={"query": "test query"})

    assert response.status_code == 504


def test_endpoint_reports_upstream_http_error():
    fake = FakeLanguageToolClient(post_results=[(500, {"error": "failure"})])
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.post("/v1/correct", json={"query": "test query"})

    assert response.status_code == 502
    assert response.json() == {"detail": "LanguageTool request failed"}


def test_endpoint_reports_invalid_upstream_json():
    fake = FakeLanguageToolClient(post_results=[(200, b"not-json")])
    install_fake_client(fake)

    with TestClient(shim.app) as client:
        response = client.post("/v1/correct", json={"query": "test query"})

    assert response.status_code == 502
    assert response.json() == {"detail": "LanguageTool request failed"}
