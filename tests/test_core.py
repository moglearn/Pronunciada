"""pronunciada.core: score_phones without a text front-end, phones_for, the target format."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

PROMPTS = {
    "en": "The river carries 8 times as much water.",
    "de": "Der Zug fährt um acht Uhr nach Berlin.",
    "es": "Tengo 31 años y vivo en Madrid.",
    "fr": "Le train part à huit heures.",
    "it": "Il treno parte alle otto per Roma.",
    "pt": "O comboio parte às oito horas.",
    "ru": "Поезд отправляется в восемь часов.",
    "ja": "電車は八時に出発します。",
    "ko": "기차는 여덟 시에 출발합니다.",
    "zh": "火车八点出发。",
}
LANGS = list(PROMPTS)
FORBIDDEN = ("num2words", "soundfile", "pyopenjtalk", "sudachipy", "sudachidict_core", "jieba", "pypinyin",
             "opencc", "pydantic")
FRONTEND_DATA = ("data/lex/", "data/cjk/")


def _target(lang):
    from pronunciada.core import phones_for

    try:
        return phones_for(PROMPTS[lang], lang)
    except ImportError as e:  # optional front-end packages
        pytest.skip(f"{lang} front-end not installed: {e}")


def _audio(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(16000 * seconds)) / 16000.0
    x = 0.1 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(len(t))
    return x.astype(np.float32)


def _same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b and type(a) is type(b)


# --------------------------------------------------------------------------- target format

@pytest.mark.parametrize("lang", LANGS)
def test_target_roundtrip_is_lossless(lang):
    from pronunciada.core import reference_from_target
    from pronunciada.text import get_g2p

    tg = _target(lang)
    tg2 = json.loads(json.dumps(tg, ensure_ascii=False))           # survives JSON text
    ref = reference_from_target(tg2, lang)
    orig = get_g2p(lang)(PROMPTS[lang])
    assert ref == orig                                             # dataclass equality, every field
    assert tg["format"] == "pronunciada-target" and tg["version"] == 1


def test_spanish_number_alternatives_travel_in_the_target():
    from pronunciada import r20
    from pronunciada.core import reference_from_target

    tg = _target("es")
    if not r20.enabled("numalt", "es"):
        pytest.skip("numalt off in this environment")
    num = [w for w in tg["words"] if any(c.isdigit() for c in w["text"])]
    assert num and num[0]["alternatives"], "31 should have candidate readings"
    ref = reference_from_target(tg, "es")
    wi = tg["words"].index(num[0])
    assert len(ref.word_alternatives[wi]) == len(num[0]["alternatives"])


def test_bad_inputs_raise():
    from pronunciada.core import reference_from_target, score_phones

    tg = {"format": "pronunciada-target", "version": 1, "lang": "es", "words": [
        {"text": "x", "syllables": [{"phones": [{"ipa": "a", "nucleus": True}]}]}]}
    assert reference_from_target(tg).phones[0].ipa == "a"
    with pytest.raises(ValueError, match="sample_rate"):
        score_phones(np.zeros(16000, np.float32), 44100, tg, "es")
    with pytest.raises(ValueError, match="1-D"):
        score_phones(np.zeros((2, 16000), np.float32), 16000, tg, "es")
    with pytest.raises(ValueError, match="float"):
        score_phones(np.zeros(16000, np.int16), 16000, tg, "es")
    bad = json.loads(json.dumps(tg))
    bad["words"][0]["syllables"][0]["phones"][0]["ipa"] = "Q"
    with pytest.raises(ValueError, match="ZIPA"):
        reference_from_target(bad)
    bad = json.loads(json.dumps(tg))
    bad["words"][0]["syllables"][0]["phones"][0]["nucleus"] = False
    with pytest.raises(ValueError, match="nucleus"):
        reference_from_target(bad)
    with pytest.raises(ValueError, match="for 'es'"):
        reference_from_target(tg, "fr")


# --------------------------------------------------------------------------- isolation (fresh process)

def run_child(job: dict, tmp_path: Path, pythonpath: Path = ROOT, env_extra=None) -> dict:
    items = []
    for k, it in enumerate(job.pop("audio_items")):
        p = tmp_path / f"a{k}.npy"
        np.save(p, it["samples"])
        items.append({"id": it["id"], "samples": str(p), "target": it["target"]})
    job["items"] = items
    jp = tmp_path / "job.json"
    jp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION",
                                                            "LINGOLIX_API_KEY")}
    env["PYTHONPATH"] = str(pythonpath)
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, "-m", "pronunciada.tools.core_child", str(jp)], capture_output=True,
                         text=True, env=env, cwd=tmp_path, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def assert_core_only(rep: dict):
    loaded = set(rep["modules"])
    assert not loaded & set(FORBIDDEN), sorted(loaded & set(FORBIDDEN))
    fe = [m for m in rep["pronunciada_modules"]
          if m.startswith("pronunciada.text.") and m not in ("pronunciada.text.base", "pronunciada.text.ipa")]
    assert not fe, fe
    bad = [p for p in rep["opened"] if any(s in p.replace(os.sep, "/") for s in FRONTEND_DATA)
           or p.endswith((".tsv.gz", ".txt.gz"))]
    assert not bad, bad


@pytest.mark.model
@pytest.mark.parametrize("lang", LANGS)
def test_score_phones_loads_no_frontend(lang, tmp_path):
    tg = _target(lang)
    rep = run_child({"lang": lang, "kw": {"threads": 1},
                     "audio_items": [{"id": "x", "samples": _audio(), "target": tg}]}, tmp_path)
    assert_core_only(rep)
    assert rep["results"]["x"]["result"]["lang"] == lang


# --------------------------------------------------------------------------- parity with the text path

@pytest.mark.model
@pytest.mark.parametrize("lang", LANGS)
def test_score_phones_equals_scorer_score(lang):
    from pronunciada.core import score_phones
    from pronunciada.scorer import Scorer

    tg = _target(lang)
    wav = _audio(seed=LANGS.index(lang))
    sc = Scorer(lang, threads=1)
    a = sc.score(wav, PROMPTS[lang])
    b = score_phones(wav, 16000, tg, lang, threads=1)
    da, db = a.to_dict(), b.to_dict()
    da.pop("timing_ms"), db.pop("timing_ms")
    assert _same(da, db)
    for s in (0.0, 1.0):
        ra, rb = sc.rescore(a, s).to_dict(), sc.rescore(b, s).to_dict()
        ra.pop("timing_ms"), rb.pop("timing_ms")
        assert _same(ra, rb)
