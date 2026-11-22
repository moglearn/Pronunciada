# Round 20: a ready-to-run head-to-head against the Lingolix API

Every Lingolix figure we have so far is the one Lingolix published on 11 October 2026. Those figures are rounded, have no intervals and name no API version. `bench/third_party/lingolix_live.py` measures Lingolix ourselves on exactly the items Pronunciada was scored on. It then puts both systems side by side, with intervals and paired differences, next to the published figure. The module needs only `LINGOLIX_API_KEY`. **The Lingolix API has not been called yet: no key exists.** Everything below was exercised against a local fake.

## Running it once the key exists

```bash
export LINGOLIX_API_KEY=...        # read from the environment only; never printed, logged or stored
python -m bench.third_party.lingolix_live --plan-only \
    --pronunciada-cache ~/.cache/pronunciada/round19/integrator/frozen_cache
python -m bench.third_party.lingolix_live --yes \
    --pronunciada-cache ~/.cache/pronunciada/round19/integrator/frozen_cache
```

The first command prints the plan and sends nothing. The second command does three things:

- sends the requests;
- caches every answer;
- writes `results/tables/lingolix_live.json` and `results/lingolix_live.md`.

To add Pronunciada's fleurs-swap test records, add `--fleurs-swap`. That covers en, fr, es, de, ja, ko, it, pt and ru: 4,050 more requests and 12.9 audio hours. `--max-requests N` caps a run. If a run is interrupted, run the same command again and it resumes.

`--pronunciada-cache` chooses which Pronunciada outputs go in the Pronunciada column. With the round-19 frozen run, the column reproduces the committed round-19 numbers: test 1 AUC en 0.977, es 0.985, de 0.967, ja 0.923, ko 0.957, and the fleurs-swap rows. Without the flag, the column uses the outputs in `$PRONUNCIADA_CACHE`, which are the 1.0.2 outputs.

`--report-only` rebuilds the table from the cache and sends nothing.

## What is sent

The only things sent are public benchmark recordings and their prompt texts:

- **Test 1:** the 1,200 Common Voice clips (CC0) of the benchmark's v2026.10 release. These are 200 each for en, fr, es, de, ja and ko. Each is sent with its manifest `target` text, and the WAV bytes are sent unchanged.
- **Test 2:** the 500 speechocean762 test clips (CC BY 4.0), with their `target` text. These are decoded from Pronunciada's lossless local copy to 16 kHz mono 16-bit WAV, the format the benchmark's `build_tier2.py` produces.
- **Optional fleurs-swap:** FLEURS test recordings (CC BY 4.0) with the original prompt and the prompts that have one word changed.

Each (recording, text) pair is one multipart POST to `https://api.lingolix.com/api/pronunciation/v3/check`. It carries the fields `sentence`, `language_code` and `speechdata`, and the header `Authorization: Bearer <key>`. This is the request the benchmark's own Lingolix adapter makes.

Requests run at 1 per second by default, one at a time; `--concurrency` allows up to 4. The 429 and 5xx answers are retried with exponential backoff and jitter, honouring Retry-After. The run stops cleanly on 401, 402, 403 and on the documented quota 429.

## Cost

These estimates come from the public price list, read on 22 November 2026. Lingolix bills audio minutes, at €0.02 to €0.05 per minute over a plan's included minutes.

| Run | Requests | Audio | Overage cost | Rounded up per request |
|---|---|---|---|---|
| Tests 1 and 2 | 1,700 | 127.7 min | €2.55 to €6.39 | €34 to €85 |
| With fleurs-swap | 5,750 | 905 min | €18 to €45 | €115 to €288 |

The last column assumes every request is billed as a whole minute. The documentation does not say whether billing rounds up each request or the monthly total. Tests 1 and 2 fit inside the Learner plan's 720 minutes either way if billing is per second. The free tier has 15 minutes and is non-commercial, so it is not enough.

## What is logged

Every raw JSON answer is saved as soon as it arrives. It goes to `$PRONUNCIADA_CACHE/third_party/lingolix_live/responses/`, one file per request, written atomically. Each file is keyed by a hash of the audio bytes, text, language and API options. Each file also holds:

- the UTC time it was received;
- the HTTP status;
- the response headers that name a version, model, request id or rate limit;
- the latency.

A cached item is never requested again unless `--refresh` is given.

Each run writes `runs/<UTC time>.json` with:

- start and end time (UTC);
- endpoint and options;
- rate settings;
- counts: planned, sent, succeeded, failed, retried, not sent, already cached;
- each failure;
- the `model_version` values the API returned, and any version header;
- the repository's git commit, with a flag for uncommitted changes;
- the benchmark version;
- the cost estimate.

The table repeats the model versions and the first and last test dates of the answers it used. If more than one model version appears, the table says so. This is what Lingolix's terms ask a published benchmark to state: the method, the test date and the model version.

## How the numbers are computed

Answers are reduced with the benchmark's Lingolix adapter, `vendors.lingolix`, which is identical at v2026.10 and at main f8750aa:

- word score = `accuracy`;
- error type from the syllables' `is_missing` and `is_extra`.

The answers are then scored with the metric code of `bench/third_party/lingolix.py`.

For this round, that file now follows `report.py` for Lingolix-shaped records too:

- **Error flag:** the expected and detected syllable IPA are compared after report.py's spelling rules for ja, fr and es. For the other languages the flag is undefined, as in report.py.
- **Sentence score:** `overall`.

The change leaves every Pronunciada metric bit-identical. Every test 1 and test 2 metric, under both rule sets, was compared before and after the change and matched exactly.

The table shows these rows:

- **Test 1:** word AUC and FA@80 per language, under the v2026.10 rules and the later main rules.
- **Japanese length errors:** how many of the 40 each system catches with its own flag.
- **Test 2:** word and sentence Pearson.
- **fleurs-swap:** far AUC, near AUC and near hit at 5% false alarms.

Each row has a 95% interval from 1,000 resamples: utterances for test 1, speakers for test 2, recordings for fleurs-swap. Each row also has the paired difference, Pronunciada minus Lingolix, on the same resamples, with a two-sided p. The published Lingolix figure is in its own column. A test that Lingolix did not fully answer is marked INCOMPLETE in its row.

## Dry run

`python -m bench.third_party.lingolix_live --dry-run --fleurs-swap` runs the whole pipeline against a fake transport: plan, rate limiter, retries, cache, resume, parsing and table. The fake answers in the documented format with random scores, and every 97th or 89th call returns a 429 or a 503. A socket guard refuses any connection or name lookup for a non-local host. Output goes only to `$PRONUNCIADA_CACHE/third_party/lingolix_live/DRYRUN_FAKE/`, under a "DRY RUN, FAKE DATA" banner.

The dry run was done in three passes:

1. Capped at 900 requests: 919 fake calls, including 19 retries.
2. Uncapped: it resumed, sent the remaining 4,850 requests (4,956 calls, including 106 retries), and re-requested none of the 900.
3. Again: 0 to send.

The fake Lingolix column is at chance (AUC about 0.5), as random scores should be. It says nothing about Lingolix.

## Assumptions not yet checked against a real answer

The benchmark repository and the public documentation leave these open. Each is checked in code (`parse_response`, `classify_status`). A real answer that breaks one stops the run and names the assumption. The answer is already cached by then, so fixing the adapter costs nothing.

- **A1 Endpoint.** `POST {LINGOLIX_BASE_URL or https://api.lingolix.com}/api/pronunciation/v3/check`. The documentation names the path and the host, but not the full URL.
- **A2 Request encoding.** multipart/form-data with `sentence`, `language_code` and the file field `speechdata`. These come from vendors.py; the documentation lists the fields but not the encoding.
- **A3 Score scale.** All scores are in [0, 1]. A 0–100 value stops the run instead of being rescaled.
- **A4 Word list.** `words` lists the sentence's words in order. Outside Japanese, at least half the sentence's words must appear in it.
- **A5 Syllable fields.** The syllable fields `expected_ipa`, `detected_ipa`, `is_missing`, `is_extra` and `is_scored` have the documented types. The benchmark's error flag depends on them.
- **A6 Model version.** The top-level `model_version` is the version string to cite. No version header is documented, so any header naming a version or model is also recorded.
- **A7 Quota 429.** A 429 is the monthly-quota error, and stops the run, if its body has `remaining_minutes`, `upgrade_url` or `plan`, or if `X-RateLimit-Remaining` is 0. Any other 429 is treated as a transient rate limit and retried. Neither a per-second limit nor Retry-After is documented.
- **A8 Billing unit.** Billing is per audio minute, measured per second, but it is not documented whether each request is rounded up. Hence the two cost figures.
- **A9 Languages and limits.** The language codes are the benchmark's, and all are in the documented list. Recordings over 60 s or 10 MiB are not sent; none of ours are.

## Tests

`tests/test_lingolix_live.py` has 23 tests and makes no network calls. They cover:

- a cache hit sends nothing a second time;
- the cache key changes when the text changes;
- resuming after Ctrl-C;
- `--max-requests`;
- backoff that honours Retry-After, in seconds and as an HTTP date, and grows exponentially on 5xx;
- clean stops on 401, 403 and quota;
- a 400 is recorded as a failure, not retried;
- a malformed answer is cached and then stops the run;
- spacing between requests;
- refusal without a key;
- the key never appears in cache files, run logs, tables or console output, even when a hostile server echoes it back;
- parsing a hand-written fixture in the documented format (`tests/data/lingolix_v3_check_response.json`) the way the benchmark's adapter does, plus one failure per assumption;
- the guard against non-local hosts;
- a dry run that must not build the real transport or open any connection.

The full suite passes: 632 tests.

## Addendum, 22 November (evening): audio-minute budget and more speechocean762 speakers

By now 1,700 answers are cached (tests 1 and 2, 129.99 audio minutes). Two facts changed the cost control. Lingolix meters by the second of audio. Paid plans do not stop at their included minutes, they bill the overage per minute, so the quota 429 no longer protects the budget.

**Ledger and cap.** `$PRONUNCIADA_CACHE/third_party/lingolix_live/ledger.jsonl` (the dry run has its own under `DRYRUN_FAKE/`) gets one line per request attempt, written and flushed before the request goes out, and one line with its outcome after it. Every attempt counts, retries and failures included. On first use it was backfilled with one line per cached response, with the seconds of the exact bytes sent: 1,700 lines, 129.99 minutes. Running the backfill again adds nothing. A live send now refuses to start without `--quota-cap-min`. `--quota-used-min` overrides the ledger total. The plan keeps only what fits under the cap. The send checks the cap again, under a lock, before every attempt, and the first attempt that would cross it stops the run. This holds at any `--concurrency`.

**More speechocean762 speakers.** `--so762-extend` sends more of the 2,500 speechocean762 test clips (125 speakers, 20 clips each). The items come from `lingolix.tier2x_items`, which applies `build_tier2.py`'s rules to Pronunciada's copy of the corpus. That copy keeps the punctuated prompt, so the target is rebuilt as the scored words joined by spaces, which is the corpus's Kaldi text. The builder reproduces all 500 benchmark items exactly, target text and human labels. The request is the one test 2 sends, so the plan over all 2,500 clips counts exactly 500 as cached. Speakers are taken whole, in ascending ID order, and the selection stops at the first speaker that does not fit.

Plan-only, from the cache as it is:

| budget | speakers | clips | audio min | next speaker (does not fit) |
|---|---|---|---|---|
| 44.5 min (`--quota-used-min 0 --quota-cap-min 44.5`) | 47 (0003 to 1420) | 743 | 43.78 | 1422: 16 clips, 1.13 min |
| 59.5 min | 62 | 989 | 58.85 | 2252: 18 clips, 0.98 min |
| cap 175 min, ledger 129.99 used | 48 (0003 to 1422) | 759 | 44.91 | 1465: 19 clips, 1.42 min |

With a cap of 175 only 0.10 minutes are left after the 759 clips. Retries use the same budget, so a few retries can stop the run inside the last speaker. That speaker then counts as incomplete and stays out of row c.

**Pronunciada's side.** `python -m bench.third_party.lingolix score --only tier2x` scored all 2,500 clips with the code at `main` a1716d3, with the same scorer settings and response shape as test 2. On the benchmark's 500 clips its answers are byte-identical to the round-19 frozen answers used by the test 2 rows (max difference 0).

**Report.** `results/lingolix_live.md` has a new section, "speechocean762, more speakers", with three rows: (a) the benchmark's 500 clips, (b) every clip Lingolix answered, and (c) the speakers with all 20 clips answered. Each row has sentence and word Pearson as test 2 computes them. Each difference has a 95% interval and a two-sided p from 10,000 paired resamples of speakers. Row a reproduces the test 2 figures: sentence 0.788 vs 0.713, word 0.581 vs 0.527, and the old p of 0.100 and 0.236. With speakers resampled, the p values are 0.074 and 0.163. The test 2 rows resample groups made of the first four digits of the utterance id, and those four digits are not the speaker ID (`0` plus the first three digits of it). So each of those groups holds up to ten speakers, and there are 83 groups for the 500 clips. The test 2 rows are left as they were. The new rows resample the corpus speaker IDs. The section also gives the cost of finishing the split: 2,000 clips, 128.77 minutes, EUR 6.44 if all of it is billed as overage at the Starter rate of EUR 0.05 per minute. A dated note records that Lingolix refused Mandarin (HTTP 400, nothing billed), so the OMPAL comparison could not be run.

**Dry run.** `--dry-run --so762-extend --quota-cap-min <dry ledger + 45> --concurrency 4` planned 48 speakers and 759 clips. The fake answered with 15 retries (429 and 503). The run made 763 attempts and stopped before the attempt that would have crossed the cap. That left the ledger 0.003 minutes under the cap. `--dry-run --report-only` then built all three rows; 47 speakers were complete. The real cache and ledger were not touched.
