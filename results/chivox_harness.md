# Chivox benchmark runner (built and tested in dry-run mode only)

`bench/third_party/chivox_live.py`, with the shared machinery in `bench/third_party/live_core.py` (sending, budget, cache, ledger) and `bench/third_party/live_eval.py` (the scoreboard metrics). Tests: `tests/test_chivox_live.py`, fake transport only. No request has been sent to Chivox. No account was created and no key was used.

## What the documentation says (read 2026-11-23, all public, no login)

Sources: https://api-portal.cloud.chivox.com/global, /docs, /pricing, /runtime, /solutions/function-calling, /faq, /products/*, /legal/terms and /legal/privacy (both updated 2026-06-01). Also the vendor's public repository https://github.com/chivox-developer/chivox-speech-eval-mcp (MIT for its docs and examples; last commit 2026-06-12), files `docs/api-reference.md` and `examples/quick-test/python/test_eval.py`.

| topic | documented |
|---|---|
| endpoint | MCP over Streamable HTTP, JSON-RPC 2.0: `POST https://mcp-global.cloud.chivox.com/` (`initialize`, `tools/list`, `tools/call`). Function-calling host `fc-global.cloud.chivox.com` (`GET /v1/functions`, `POST /v1/call`), but its request body is not documented, so the runner does not use it. Streaming WebSockets exist and are not used. |
| authentication | `Authorization: Bearer <api key>` on every request. One key, no secret and no signature on the self-serve (B2C) path. The repository also describes a B2B JWT scheme (`cvx_ak_` / `cvx_sk_`, HS256), which the portal does not offer to this account. |
| tools used | `en_sentence_eval` (English read-aloud sentence) and `cn_sentence_eval` (Mandarin sentence). Arguments: `ref_text`, plus one of `audio_base64` or `audio_url`. Optional: `accent` (1 British, 2 American, 3 neutral, default 3), `rank` (4 or 100, default 100), `voiced` (en, default 1 strict), `age_group` (zh, `child` or `adult`, default adult), `attachAudioUrl` (default 0). |
| Mandarin text | Only Chinese characters appear in the examples (`"今天天气真好"`, Simplified). No pinyin input is documented for `cn_sentence_eval`. |
| audio | "MP3, WAV, etc.", recommended 16 kHz mono; the FAQ lists mp3 / wav / m4a / ogg / aac / pcm. Max 50 MB per file. No maximum duration is published. |
| response | The scoring JSON is the text of `result.content[0]`. Fields: `overall` ("weighted roll-up"), `accuracy` ("phoneme / syllable correctness"), `pron` ("articulation quality"), `integrity` ("did the user read every word?"), `fluency.overall` / `.speed` / `.pause`, `details[i].char` / `.score`, `details[i].phone[j].phoneme` / `.score` / `.dp_type`, `phoneme_error {expected, actual}` (English), and `tone_ref` / `tone_detected` (Mandarin). Scale 0–100. No version field. |
| errors | 400 malformed; 401 missing or bad token; 403 "API Key invalid, disabled, or quota exhausted"; 404 session not found; 409 invalid state; a "structured 429" when a key's cap is reached (/runtime); JSON-RPC error `-32000` "Evaluation failed: …". |
| limits | /docs "technical guardrails": 3 parallel scoring workers, 10 pending jobs per key, 50 MB per upload, audio TTL 5 min. No rate limit (QPS) is published. |
| billing | Points: 1 point per word, phrase or sentence evaluation; 2 per paragraph; 0 for a failed call. Free: 600 points, valid 30 days, no card. Paid, per evaluation: Standard $0.0038, Advanced $0.0035 (+10 % bonus points), Flagship $0.0032 (+20 %), packs from $19.90 / $99.90 / $199.90. Points expire 30 days after they land (/pricing). The terms instead say "as stated in the console", so the two disagree. |
| retention | /docs: "Audio is processed, scored, and dropped. We retain the resulting JSON for 30 days." /runtime: "zero audio retention", streaming TTL 0 s. /docs Limits: a 500 MB "temporary audio cache" and a 5-minute audio TTL. Privacy policy: content is kept "as long as needed to provide the Service and configured retention settings". Processing in the United States "and other countries". Region-locked tenants (US, EU, SG) only on enterprise plans. |
| training | Privacy: "We do not use your evaluation content to train public models unless you opt in to a separate program disclosed in the console." Terms §7 grants a licence to process inputs and outputs "to provide and improve the Service". |
| benchmarking | Neither the terms nor the privacy policy restricts benchmarking or publishing comparisons. Terms §9: scores are "informational and not a guarantee of accuracy". Terms §4 forbids processing children's voice data without a lawful basis (none of the audio here is children's speech). |

The public documentation was enough to build the client: single Bearer key, endpoint, arguments, response fields and billing unit. Two points can only be settled by the first real answer. The runner checks both and stops on a mismatch.

1. **Whether the MCP server accepts a 16 kHz WAV in `audio_base64` without a format argument.** No `audio_format` is documented for the one-shot tools.
2. **The exact wrapper.** The docs' Quickstart nests the result once more (`result.details`). The parser accepts both shapes.

Run the 2-point smoke test (below) first.

## Field mapping (fixed from the documentation before any response existed)

| level | speechocean762 row | OMPAL row | Chivox field |
|---|---|---|---|
| sentence total | total | — | `overall` |
| sentence accuracy | accuracy | accuracy | `accuracy` |
| sentence fluency | fluency | fluency | `fluency.overall` |
| sentence prosodic | prosodic: no Chivox score ("—") | prosody: no Chivox score ("—") | none documented |
| secondary | vs expert accuracy | vs teacher accuracy (the slot Azure's PronScore takes in results/ompal.md) | `pron` (en), `overall` (zh) |
| completeness | JSON only | JSON only | `integrity` |
| word | word Pearson, low-word AUC | word unit (one character) | `details[i].score` |
| sound | sound Pearson | sounds channel = min phone score of the unit | `details[i].phone[j].score` |
| tone | — | tone channel: 100 if `tone_ref` = `tone_detected`, else 0 | `tone_ref`, `tone_detected` |

The brief asked for sentence total mapped to overall and sentence accuracy mapped to pronunciation / accuracy. Chivox documents both `accuracy` ("phoneme / syllable correctness") and `pron` ("articulation quality"). Accuracy is mapped to `accuracy`, which is the closer definition of the experts' accuracy score. `pron` is reported as the secondary row.

**Alignment.**
- English words: the human words are lined up with `details[].char` by the Lingolix benchmark's word alignment (`lingolix.align`). A human word with no Chivox word gets 0, the lowest score, and is counted.
- English phones: each human phone (ARPAbet mapped to IPA) is aligned with that word's `phone[]` using `bench.eval_azure_so762.align`, the same alignment the Azure rows use. An unaligned phone takes its word's score and is counted.
- Mandarin: each `details[i].char` is located left to right in the Simplified prompt. Because the conversion is one character for one character, its span is the OMPAL unit span.

## What is sent

- **speechocean762 TEST**: all 2,500 clips (Apache-2.0). Prompt = the scored words joined by single spaces (`lingolix.tier2x_items`, the convention of the Lingolix all-2,500 run), so the human word labels line up. `en_sentence_eval` with `accent` 3 and `rank` 100, both the documented defaults sent explicitly.
- **OMPAL TEST**: 745 clips including the 3 native speakers (CC BY 4.0; `bench.eval_ompal.records("test")`). `cn_sentence_eval` with `ref_text` in Simplified characters. The conversion uses OpenCC t2s, the converter of the Azure zh-CN run; the corpus prompts are Traditional, and Chivox documents only Simplified examples. `age_group` is `adult`, `rank` is 100.
- **Audio**: decoded locally to 16 kHz mono PCM16 WAV. The conversion is deterministic, so the same file always gives the same bytes and the same cache key. It is sent inline as base64.
- **Nothing else.** Every other set is refused, including L2-ARCTIC, EpaDB, Speech Accent Archive, Heroico, LeaP and Common Voice.

## Safety

- **Dry run is the default.** A live send needs `--yes`, `--points-cap N` and `CHIVOX_API_KEY`. Without all three it prints the plan and sends nothing.
- **The cap.** Every attempt books 1 point under a lock before it is sent, retries included. Points already in the ledger count, and the first refusal ends the run. `initialize` is not an evaluation; it is logged at 0 points.
- **Ledger**: `$PRONUNCIADA_CACHE/third_party/chivox_live/ledger.jsonl`. Each attempt line (UTC time, set, item id, request hash, cost, unit) is fsynced before sending. Each result line records the HTTP status.
- **Cache**: every answer is written atomically the moment it arrives, to `responses/<h[:2]>/<h>__<item>__<server name-version>.json`. Here `h` hashes the audio, text, language and request options. An answered or finally rejected request is never sent again; `--retry-rejected` overrides this for rejections only.
- **Stopping**: the run stops at once on 401, 402, 403, any 429 without Retry-After, or a JSON-RPC error that mentions quota, points, balance, credits or the key. At most 4 attempts with exponential backoff, for 5xx, 408, timeouts and "upstream" or "capacity" messages only. 404 (session expired) re-initializes and retries. Requests go out at most one every 2 s, one at a time.
- **The key** is read only from `CHIVOX_API_KEY`. The transport refuses any host but `mcp-global.cloud.chivox.com`. Every file written is scrubbed for the key, and the Authorization header is never stored.

## Cost (from the real item lists)

| run | requests | points | USD at paid rates (Flagship to Standard) |
|---|---|---|---|
| smoke test (`--limit 1`) | 2 | 2 | — (free points) |
| pilot (`--pilot`: 300 speechocean762 + 300 OMPAL) | 600 | 600 | — (exactly the 600 free points) |
| full (2,500 + 745) | 3,245 | 3,245 | $10.38 to $12.33 |

The pilot sample is seeded (seed 0) and speaker-balanced: one clip per speaker per round, in a seeded speaker order. It covers all 125 speechocean762 speakers and all 21 OMPAL speakers. A smaller `--limit` is a prefix of the same order, so the smoke test's 2 items count towards the pilot.

Because every retry also books a point, a pilot capped at 600 stops early if anything is retried. Chivox does not charge failed calls, so in that case raise the cap by the number of retries the ledger shows, without buying anything. After the free points, the full run needs about 2,650 more points: a Standard pack plus a top-up, or an Advanced pack.

## Commands

```
export CHIVOX_API_KEY=...        # the user's key; never in a file
python -m bench.third_party.chivox_live --plan                              # full run: requests, points, cached
python -m bench.third_party.chivox_live --pilot --plan                      # pilot plan
python -m bench.third_party.chivox_live --limit 1 --yes --points-cap 2      # smoke test: 1 English + 1 Mandarin
python -m bench.third_party.chivox_live --pilot --yes --points-cap 600      # pilot
python -m bench.third_party.chivox_live --pilot --report-only               # results/chivox_live.md + results/tables/chivox_live.json
python -m bench.third_party.chivox_live --yes --points-cap 3245             # full run (resumes; cached items are skipped)
python -m bench.third_party.chivox_live --report-only                       # full report
python -m bench.third_party.chivox_live --dry-run --pilot                   # rehearsal against the local fake (no network)
```

Run all of them from the repository root with the project virtualenv.

## Evaluation (`--report-only`)

The report uses only the items Chivox answered.

- **Pronunciada.** On speechocean762, the scoreboard's own outputs (`bench.eval_english.so762_items` over the cached `api_run` pickle; sound model B, `word_reg`). On OMPAL, the latest `bench.eval_ompal` ours run (the report names the cache directory).
- **Azure.** Cached responses only: speechocean762 `aligned_test.jsonl` and OMPAL zh-TW.
- **speechocean762 metrics.** Pearson with the experts for sounds, words and the sentence (accuracy, total, fluency, prosodic). Low-word AUC: expert word accuracy ≤ 6 (build_tier2.py's "mispronounced") against the word score.
- **OMPAL metrics.** The rows of results/ompal.md: word AUC (any error, tone-only, consonant or vowel; all speakers and learners only), the tone and sounds channels, hit rate at 5 % and 10 % false alarms (native or learner thresholds), and learner sentence Pearson and Spearman.
- **Intervals.** 10,000 speaker resamples with every system on the same resamples, giving Chivox − Pronunciada with a 95 % interval and a two-sided p. The hit rates use 2,000 resamples.
- **Counts.** Answered, rejected and failed requests, ledger attempts and HTTP statuses.
- **Samples.** A pilot report says on each set that it is a sample of n of N clips and that the intervals describe that sample.
- **Output.** `results/chivox_live.md` and `results/tables/chivox_live.json`. A dry run writes under `DRYRUN_FAKE/` with a "DRY RUN, FAKE DATA" banner, never to `results/`.
- **Cost of the report.** About 90 s on one core for the pilot.

## Show the user before any audio is sent

1. **Training opt-in.** The privacy policy mentions "a separate program disclosed in the console" for training on evaluation content. Ask the user to check the console and leave it off (opt-in, so off by default).
2. **Retention.** Results JSON is kept 30 days per /docs; the privacy policy refers to "configured retention settings". Ask whether the console has a retention setting, and set it to the shortest.
3. **Region.** No region choice on self-serve plans; data may be processed in the US "and other countries". The audio is public research corpora (speechocean762 Apache-2.0, OMPAL CC BY 4.0), no user data.
4. **Points expiry.** The free 600 points expire 30 days after they land.
5. **Status of the service.** The /docs changelog says the service is "invite-only" and "ships from local builds", while /global offers self-serve sign-up. The user's account and key exist, so this is a note only.
6. **Key caps.** /runtime mentions scoped keys and per-key monthly caps. Setting a cap of 600 points on the key in the console gives a second, server-side limit.

Environment variable: **`CHIVOX_API_KEY`** (the single key the console issued). The documented self-serve authentication needs nothing else: no app id, secret, region or product id.
