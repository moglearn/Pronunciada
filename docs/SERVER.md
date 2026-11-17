# Local HTTP server

> **Development server.** `pronunciada serve` has no authentication, no TLS and no rate
> limiting. It binds to `127.0.0.1` by default; do not bind it to a public interface or expose
> it to a network you do not trust. Put a real web server / gateway with authentication in
> front of it if other machines need it.

Standard library only (`http.server`), no extra dependency.

```
pronunciada serve [--host 127.0.0.1] [--port 8765] [--azure-compat] [--max-mb 20] [--threads N]
pronunciada serving on http://127.0.0.1:8765 (POST /v1/score, Azure-format path enabled); Ctrl-C to stop
```

Requests are scored one at a time (a lock around the model), one `Scorer` is kept per
language after its first request, and request bodies above `--max-mb` (default 20 MB) get 413.

## Endpoints

| method, path | |
|---|---|
| `GET /healthz` | `{"status": "ok", "version": "0.1.0", "azure_compat": true}` |
| `GET /v1/languages` | `{"languages": [...]}` as `pronunciada langs --json` |
| `POST /v1/score` | score one recording → the `pronunciada check --json` document |
| `POST /speech/recognition/{conversation,interactive,dictation}/cognitiveservices/v1` | only with `--azure-compat`: Azure-shaped result ([AZURE_COMPAT.md](AZURE_COMPAT.md)) |

`POST /v1/score` takes the audio either as the raw body (any `Content-Type` except form
encodings) or as `multipart/form-data` with a file field `audio`. Fields, as query parameters
or form fields:

| field | default | |
|---|---|---|
| `text` | required | the expected text |
| `lang` | `en` | code or alias; a locale such as `en-US` or `es-ES` falls back to its language |
| `strictness` | 0.5 | 0 … 1 |
| `coach` | 0 | 1 adds the `coaching` block (items + tips) |
| `sensitivity` | balanced | careful / balanced / eager |
| `max_tips` | 3 | |

Audio: anything libsndfile reads (WAV, FLAC, OGG), any sample rate, mono or stereo.

## Examples

```
$ curl -s localhost:8765/healthz
{"status": "ok", "version": "0.1.0", "azure_compat": true}

$ curl -s -X POST "http://127.0.0.1:8765/v1/score?text=The%20quick%20brown%20fox%20jumps%20over%20the%20lazy%20dog&lang=en" \
       -H "Content-Type: audio/wav" --data-binary @clip.wav
{"lang": "en", "text": "The quick brown fox jumps over the lazy dog", "prob": 0.987..., "words": [...], ...}

$ curl -s -F audio=@clip.wav --form-string "text=The quick brown cat jumps over the lazy dog" \
       --form-string lang=en --form-string coach=1 http://127.0.0.1:8765/v1/score
{..., "prob": 0.911..., "coaching": {"schema_version": 2, "items": [...],
 "tips": [{"tip_key": ..., "text": "The word “cat” seemed to be missing; try saying every word.", ...}, ...]}}

$ curl -s -X POST "http://127.0.0.1:8765/v1/score?lang=en" -H "Content-Type: audio/wav" --data-binary @clip.wav
{"error": "text is required"}                                   (HTTP 400)
```

## Errors

Always JSON `{"error": "..."}`:

| status | when |
|---|---|
| 400 | missing text, unsupported language, strictness outside 0–1, audio that cannot be decoded, bad Azure header |
| 404 | unknown path (the Azure path without `--azure-compat`) |
| 405 | wrong method |
| 411 | POST without `Content-Length` or chunked encoding |
| 413 | body larger than `--max-mb` |
| 422 | the scorer rejected the input (for example no scorable sounds in the text) |
| 501 | the language's optional extra is not installed (`pip install 'pronunciada[zh]'`) |
| 500 | anything else (the message names only the exception type) |

The access log on stderr has the request line only; request headers (where an Azure client
puts its subscription key) and bodies are never logged.

## Embedding

```python
from pronunciada.server import make_server
srv = make_server("127.0.0.1", 0, azure_compat=True)     # port 0: pick a free port
port = srv.server_address[1]
# threading.Thread(target=srv.serve_forever, daemon=True).start(); ...; srv.shutdown()
```

This is how `tests/test_entrypoints.py` runs it.
