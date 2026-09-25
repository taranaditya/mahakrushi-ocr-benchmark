from scripts.score_final_test import ocrd_character_counts, ocrd_graphemes


def normalized_cer(errors: int, correct: int) -> float:
    return errors / (errors + correct) if errors + correct else 0.0


def test_devanagari_letter_and_matra_are_one_grapheme():
    assert ocrd_graphemes("कि") == ["कि"]
    errors, correct, reference_units = ocrd_character_counts("कि", "क")
    assert (errors, correct, reference_units) == (1, 0, 1)
    assert normalized_cer(errors, correct) == 1.0


def test_nfc_equivalent_text_has_zero_error():
    assert ocrd_character_counts("क़", "क़") == (0, 1, 1)


def test_insertions_are_counted_and_normalized_cer_is_bounded():
    errors, correct, reference_units = ocrd_character_counts("a", "aaaa")
    assert (errors, correct, reference_units) == (3, 1, 1)
    assert errors / reference_units == 3.0
    assert normalized_cer(errors, correct) == 0.75


def test_case_and_punctuation_remain_real_transcription_errors():
    assert ocrd_character_counts("A,", "a.") == (2, 0, 2)


def test_bidirectional_format_controls_are_ignored():
    assert ocrd_character_counts("text", "te\u200fxt") == (0, 4, 4)
