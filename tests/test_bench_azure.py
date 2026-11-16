"""Parser tests against the response examples in Azure's documentation (no live calls)."""
import base64
import json

import pytest

from bench import azure_pa as A

# REST API for short audio, "typical response for recognition with pronunciation assessment"
# (learn.microsoft.com/azure/ai-services/speech-service/rest-speech-to-text-short), trimmed.
REST = {
    "RecognitionStatus": "Success", "Offset": 700000, "Duration": 8400000,
    "DisplayText": "Good morning.", "SNR": 38.76819,
    "NBest": [{
        "Confidence": 0.98503506, "Lexical": "good morning", "ITN": "good morning",
        "MaskedITN": "good morning", "Display": "Good morning.",
        "AccuracyScore": 100.0, "FluencyScore": 100.0, "ProsodyScore": 87.8,
        "CompletenessScore": 100.0, "PronScore": 95.1,
        "Words": [
            {"Word": "good", "Offset": 700000, "Duration": 2600000, "Confidence": 0.0,
             "AccuracyScore": 100.0, "ErrorType": "None",
             "Feedback": {"Prosody": {"Break": {"ErrorTypes": ["None"], "BreakLength": 0},
                                      "Intonation": {"ErrorTypes": [], "Monotone": {
                                          "Confidence": 0.0, "WordPitchSlopeConfidence": 0.0,
                                          "SyllablePitchDeltaConfidence": 0.91385907}}}}},
            {"Word": "morning", "Offset": 3400000, "Duration": 5700000, "Confidence": 0.0,
             "AccuracyScore": 100.0, "ErrorType": "None",
             "Feedback": {"Prosody": {"Break": {"ErrorTypes": ["None"],
                                                "UnexpectedBreak": {"Confidence": 3.5294118e-08},
                                                "MissingBreak": {"Confidence": 1.0},
                                                "BreakLength": 0},
                                      "Intonation": {"ErrorTypes": [], "Monotone": {
                                          "Confidence": 0.0, "WordPitchSlopeConfidence": 0.0,
                                          "SyllablePitchDeltaConfidence": 0.91385907}}}}},
        ]}],
}


def _ph(p, acc, off, dur, nb):
    return {"Phoneme": p, "PronunciationAssessment": {"AccuracyScore": acc, "NBestPhonemes": [
        {"Phoneme": a, "Score": s} for a, s in nb]}, "Offset": off, "Duration": dur}


# Speech SDK JSON (how-to-pronunciation-assessment, "spoken phoneme" example), IPA, nested layout.
SDK = {
    "Id": "bbb42ea51bdb46d19a1d685e635fe173", "RecognitionStatus": 0, "Offset": 7500000,
    "Duration": 13800000, "DisplayText": "Hello.",
    "NBest": [{
        "Confidence": 0.975003, "Lexical": "hello", "ITN": "hello", "MaskedITN": "hello",
        "Display": "Hello.",
        "PronunciationAssessment": {"AccuracyScore": 100, "FluencyScore": 100,
                                    "CompletenessScore": 100, "PronScore": 100},
        "Words": [{
            "Word": "hello", "Offset": 7500000, "Duration": 13800000,
            "PronunciationAssessment": {"AccuracyScore": 99.0, "ErrorType": "None"},
            "Syllables": [
                {"Syllable": "hɛ", "PronunciationAssessment": {"AccuracyScore": 91.0},
                 "Offset": 7500000, "Duration": 4100000},
                {"Syllable": "loʊ", "PronunciationAssessment": {"AccuracyScore": 100.0},
                 "Offset": 11700000, "Duration": 9600000}],
            "Phonemes": [
                _ph("h", 98.0, 7500000, 3500000, [("h", 100.0), ("oʊ", 52.0), ("ə", 35.0), ("k", 23.0), ("æ", 20.0)]),
                _ph("ɛ", 47.0, 11100000, 500000, [("ə", 100.0), ("l", 52.0), ("ɛ", 47.0), ("h", 17.0), ("æ", 2.0)]),
                _ph("l", 100.0, 11700000, 1100000, [("l", 100.0), ("oʊ", 46.0), ("ə", 5.0), ("ɛ", 3.0), ("u", 1.0)]),
                _ph("oʊ", 100.0, 12900000, 8400000, [("oʊ", 100.0), ("d", 29.0), ("t", 24.0), ("n", 22.0), ("l", 18.0)]),
            ]}]}],
}


def test_parse_rest_flat_layout():
    p = A.parse_response(REST)
    assert p["status"] == "Success"
    s = p["sentence"]
    assert (s["AccuracyScore"], s["FluencyScore"], s["ProsodyScore"], s["CompletenessScore"], s["PronScore"]) \
        == (100.0, 100.0, 87.8, 100.0, 95.1)
    assert [w["word"] for w in p["words"]] == ["good", "morning"]
    w = p["words"][1]
    assert w["accuracy"] == 100.0 and w["error_type"] == "None"
    assert w["start"] == pytest.approx(0.34) and w["dur"] == pytest.approx(0.57)
    assert w["missing_break_conf"] == 1.0 and w["unexpected_break_conf"] == pytest.approx(3.53e-8, rel=1e-3)


def test_parse_sdk_nested_layout_with_phonemes():
    p = A.parse_response(SDK)
    assert p["sentence"]["PronScore"] == 100.0 and p["sentence"]["ProsodyScore"] is None
    w = p["words"][0]
    assert w["accuracy"] == 99.0
    assert [x["phoneme"] for x in w["phones"]] == ["h", "ɛ", "l", "oʊ"]
    assert w["phones"][1]["accuracy"] == 47.0 and w["phones"][1]["nbest"][0] == ("ə", 100.0)
    assert [x["syllable"] for x in w["syllables"]] == ["hɛ", "loʊ"]
    assert w["syllables"][1]["start"] == pytest.approx(1.17)


def test_align_words_with_miscues():
    resp = json.loads(json.dumps(REST))
    words = resp["NBest"][0]["Words"]
    words.insert(1, {"Word": "uh", "AccuracyScore": 0, "ErrorType": "Insertion"})
    words.append({"Word": "everyone", "AccuracyScore": 0, "ErrorType": "Omission"})
    fl = A.flatten(resp, ["Good", "morning,", "everyone"])
    assert [w["accuracy"] for w in fl["words"]] == [100.0, 100.0, 0.0]
    assert fl["words"][2]["omitted"] and not fl["words"][0]["omitted"]
    assert fl["words"][0]["azure_words"] == ["good"]


def test_align_cjk():
    resp = {"RecognitionStatus": "Success", "NBest": [{"Lexical": "你好 世界", "AccuracyScore": 80,
            "Words": [{"Word": "你好", "AccuracyScore": 90, "ErrorType": "None"},
                      {"Word": "世界", "AccuracyScore": 40, "ErrorType": "Mispronunciation"}]}]}
    fl = A.flatten(resp, ["你", "好", "世界"])
    assert [w["accuracy"] for w in fl["words"]] == [90.0, 90.0, 40.0]
    assert fl["words"][2]["error_types"] == ["Mispronunciation"]


def test_failed_recognition():
    p = A.parse_response({"RecognitionStatus": "NoMatch"})
    assert p["sentence"] is None and p["words"] == []
    fl = A.flatten({"RecognitionStatus": "InitialSilenceTimeout"}, ["a", "b"])
    assert all(w["omitted"] and w["accuracy"] is None for w in fl["words"])


def test_params_and_header():
    p = A.pa_params("Hello.", "en-US")
    assert p["PhonemeAlphabet"] == "IPA" and p["EnableProsodyAssessment"] == "True"
    q = A.pa_params("こんにちは", "ja-JP")
    assert "PhonemeAlphabet" not in q and "EnableProsodyAssessment" not in q
    hdr = base64.b64encode(json.dumps(q, ensure_ascii=False).encode()).decode()
    assert json.loads(base64.b64decode(hdr))["ReferenceText"] == "こんにちは"
    assert set(A.LOCALES.values()) >= {"en-US", "zh-CN", "ja-JP", "es-MX", "fr-FR", "de-DE",
                                       "it-IT", "pt-BR", "ru-RU", "ko-KR"}


def test_missing_credentials_exit(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
    with pytest.raises(SystemExit) as e:
        A.credentials()
    assert "AZURE_SPEECH_KEY" in str(e.value)


def test_send_retries_and_never_leaks_key(monkeypatch, capsys):
    calls = []

    class R:
        def __init__(self, code, body=None):
            self.status_code, self._b, self.headers, self.text = code, body, {"Retry-After": "0"}, ""

        def json(self):
            return self._b

    class S:
        def post(self, url, headers, data, timeout):
            calls.append(headers)
            return R(429) if len(calls) < 2 else R(200, REST)

    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    out = A.send(S(), "SECRET123", "westus", "en-US", A.pa_params("Good morning.", "en-US"), b"x")
    assert out["NBest"][0]["PronScore"] == 95.1 and len(calls) == 2
    cap = capsys.readouterr()
    assert "SECRET123" not in cap.out + cap.err
