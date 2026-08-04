# SearXNG query-corrector shim

A conservative compatibility service between the custom SearXNG `query_corrector` engine and a self-hosted LanguageTool server.

## Behavior

The shim:

- accepts `POST /v1/correct` with `query` and optional `language`;
- calls LanguageTool's `POST /v2/check`;
- considers only matches with `rule.issueType == "misspelling"`;
- rejects case-only changes, technical-looking tokens, SearXNG query-syntax tokens, unsafe control characters, whitespace replacements by default, excessive edit distances, genuinely overlapping edits, and queries requiring too many edits;
- deduplicates repeated LanguageTool matches that refer to the same source span;
- preserves the original token's lowercase, uppercase, or title-case pattern when applying a replacement;
- caches LanguageTool-supported languages through `/readyz` and falls back to auto-detection for unsupported language codes;
- permits insertion of the zero-width joiner and non-joiner characters that SearXNG accepts for scripts that require them, while suppressing deletion-only changes;
- returns `{"correction": null}` whenever it cannot make a conservative correction;
- exposes `/healthz` and `/readyz`.

## Chosen LanguageTool image

```yaml
image: docker.io/erikvl87/languagetool:6.8
```

## Build and publish the shim

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag $DOCKER_HUB/$USER_NAME/searxng-query-corrector:v1.0 \
  --tag $DOCKER_HUB/$USER_NAME/searxng-query-corrector:latest \
  --push \
  .
```

## Compose integration

Merge the `languagetool` and `query-corrector` services from `compose.example.yml` into the existing SearXNG Compose file.

No host ports are required. Docker Compose DNS makes these internal addresses available:
- `http://languagetool:8010`
- `http://query-corrector:8000`

Then configure SearXNG:

```yaml
use_default_settings: true

engines:
  - name: query corrector
    base_url: http://query-corrector:8000
    enable_http: true
    timeout: 1.0
    inactive: false
    disabled: false
```

The engine rejects plain HTTP unless `enable_http: true` is set. This is appropriate for the private Docker network shown above; use HTTPS for a remote service.

The bundled SearXNG engine timeout is `1.0` second. Keep `LANGUAGETOOL_TIMEOUT` below it so the shim can return cleanly before SearXNG cancels its request; the example uses `0.8` seconds. The shim enforces this as a hard wall-clock budget across connection setup, response streaming, and the optional `language=auto` retry.

The Docker health check deliberately calls `/readyz`, not `/healthz`. The container is therefore marked unhealthy while LanguageTool is unavailable, even when the shim process itself is still serving requests.

`/readyz` refreshes the cache of LanguageTool-supported language codes. If that probe is not called, the shim still works and falls back to a single `language=auto` retry when LanguageTool rejects a supplied code.

The example runs LanguageTool with a read-only root filesystem and `/tmp` mounted as tmpfs. Confirm this remains compatible when changing the image or its configuration; the shim readiness check exposes startup and reachability failures.

## Direct test

```yaml
query-corrector:
  ports:
    - "127.0.0.1:8000:8000"
```

Then:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{"query":"search querry","language":"en-US"}' \
  http://127.0.0.1:8000/v1/correct
```

Expected:

```json
{"correction":"search query"}
```

## Environment variables

| Variable | Default | Purpose |
|---|---:|---|
| `LANGUAGETOOL_URL` | `http://languagetool:8010` | LanguageTool base URL |
| `LANGUAGETOOL_TIMEOUT` | `0.8` | Upstream timeout in seconds |
| `DEFAULT_LANGUAGE` | `auto` | Used when SearXNG sends no language |
| `PREFERRED_VARIANTS` | `en-US,de-DE,pt-PT` | LanguageTool auto-detection variants |
| `LANGUAGE_VARIANTS` | `en:en-US,de:de-DE,pt:pt-PT` | Variantless-language mappings |
| `IGNORED_WORDS` | empty | Comma-separated technical terms |
| `MAX_QUERY_LENGTH` | `80` | Maximum accepted query length |
| `MAX_CORRECTION_LENGTH` | `256` | Maximum returned correction length |
| `MAX_LANGUAGETOOL_RESPONSE_BYTES` | `1048576` | Maximum content-decoded response body accepted from LanguageTool |
| `MAX_EDITS` | `2` | Maximum spelling edits per query |
| `MAX_TOKEN_EDIT_DISTANCE` | `2` | Maximum edit distance per token |
| `MIN_TOKEN_LENGTH` | `3` | Ignore shorter source tokens; replacement words may be shorter |
| `ALLOW_WHITESPACE_REPLACEMENTS` | `false` | Permit multiword suggestions while still requiring every part to be a plain word |
| `LOG_LEVEL` | `WARNING` | Shim log level: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, or `DEBUG` |

## Tests

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
pytest -q
```

## License

This project is released under the Zero-Clause BSD license (`0BSD`).
Attribution is not required.
