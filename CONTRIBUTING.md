# Contributing to Pronunciada

Thanks for taking the time. This project has one rule that matters more than any style guide:
**a change to scoring is judged by measurement, on data it was not tuned on.**

## Setting up

```bash
git clone https://github.com/moglearn/pronunciada
cd pronunciada
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest -q
```

The first run downloads the speech model (about 250 MB) into `~/.cache/pronunciada`.

## Kinds of contribution

**Bug reports.** Include the language, the expected text, what you got and what you expected.
A short recording you are allowed to share helps more than anything else.

**A new language.** You need a text front-end that turns text into expected sounds
(`pronunciada/text/`), a small output adapter trained on native speech, and a calibration
offset. `docs/DESIGN.md` describes each piece. Open an issue first so we can agree on the
sound inventory.

**Scoring changes.** Follow the pattern used in every round under `results/`:

1. State what you expect to improve and the bar it has to clear, before measuring.
2. Select on development data (training splits, FLEURS validation, the dev halves).
3. Run the held-out benchmarks once and report the numbers whatever they are.
4. Check that native speakers are not marked down (the guards in `bench/calib/`).

Changes that fail their bar are still worth a write-up; several of ours are in `results/`.

**Docs, examples, integrations.** Always welcome. Keep examples runnable.

## Pull requests

- Keep them focused; one change per pull request.
- `pytest -q` must pass. Scores for existing languages must stay byte-identical unless the
  pull request is a scoring change with the measurements above.
- Do not add recordings or third-party data to the repository. Point to the source and its
  licence instead, and add it to `docs/LICENSING.md`.
- New dependencies at runtime need a strong reason: the engine runs on numpy and ONNX Runtime.

## Licence

By contributing you agree that your contribution is licensed under Apache-2.0.
