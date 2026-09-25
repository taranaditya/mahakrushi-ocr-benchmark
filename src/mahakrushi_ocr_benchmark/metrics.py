"""Literal, Unicode-safe OCR metrics. No transliteration or correction."""
from .contracts import CriticalFieldScore, DocumentMetrics, DocumentRecord, OCRPrediction
from .normalization import normalize_nfc, normalize_whitespace


def _distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, 1):
        current = [i]
        for j, right_value in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_value != right_value)))
        previous = current
    return previous[-1]

def character_error_rate(reference: str, prediction: str) -> float:
    reference, prediction = normalize_nfc(reference), normalize_nfc(prediction)
    return 0.0 if not reference else _distance(list(reference), list(prediction)) / len(reference)

def word_error_rate(reference: str, prediction: str) -> float:
    reference_words, prediction_words = normalize_whitespace(reference).split(), normalize_whitespace(prediction).split()
    return 0.0 if not reference_words else _distance(reference_words, prediction_words) / len(reference_words)

def score_critical_fields(expected: dict[str, str | None], predicted: str | dict[str, str | None]) -> CriticalFieldScore:
    structured = predicted if isinstance(predicted, dict) else {}
    text = "" if isinstance(predicted, dict) else predicted
    score = CriticalFieldScore(total_expected=sum(value is not None for value in expected.values()))
    for key, value in expected.items():
        actual = structured.get(key)
        if value is None:
            if actual not in (None, ""):
                score.hallucinated.append(key)
        elif actual == value or (not structured and value in text):
            score.exact_matches += 1
        else:
            score.missing.append(key)
    return score

def score_prediction(record: DocumentRecord, prediction: OCRPrediction) -> DocumentMetrics:
    reference = record.ground_truth.read_text(encoding="utf-8")
    output = prediction.raw_text
    whitespace_reference, whitespace_output = normalize_whitespace(reference), normalize_whitespace(output)
    fields: str | dict[str, str | None] = prediction.structured_output or output
    return DocumentMetrics(model_id=prediction.model_id, document_id=record.id, cer_strict=character_error_rate(reference, output), cer_whitespace=character_error_rate(whitespace_reference, whitespace_output), wer_strict=word_error_rate(reference, output), wer_whitespace=word_error_rate(whitespace_reference, whitespace_output), field_score=score_critical_fields(record.critical_fields, fields))
