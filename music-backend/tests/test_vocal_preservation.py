"""Regression checks for audible syllables lost by Songscription post-processing."""

from __future__ import annotations

import unittest

import numpy as np

from cover_studio.lyrics_schema import classify_vocal_event, normalize_vocable_elongation
from cover_studio.lyrics_service import (
    HALLUCINATION_PATTERNS,
    _align_vocalise_to_envelope,
    _clean_vocalise_text,
    _is_runaway_token_loop,
    _validate_words,
    sanitize_runaway_repetition,
)
from cover_studio.music_asr_engine import _enforce_non_overlapping_same_stream


class VocalPreservationTests(unittest.TestCase):
    def test_repeated_syllables_and_scat_patterns_keep_their_count(self):
        for phrase in (", ".join(["La"] * 12), "doo-ba-da-da-da", " ".join(["Ba ee ya"] * 5)):
            with self.subTest(phrase=phrase):
                cleaned, is_adlib = _clean_vocalise_text(phrase, "en", allow_rhythmic=True)
                self.assertEqual(cleaned, phrase)
                self.assertTrue(is_adlib)
                self.assertFalse(_is_runaway_token_loop(phrase))
        self.assertEqual(classify_vocal_event("shoo-bap-doo")[0], "vocable")

    def test_normalization_preserves_hyphenated_articulation(self):
        self.assertEqual(normalize_vocable_elongation("Ba-i-i-e-e-e-e"), "Ba-i-i-e-e-e-e")
        self.assertEqual(normalize_vocable_elongation("Ooooooooooh"), "Ooh")
        self.assertIsNone(_clean_vocalise_text("\u266a (Vocalise)", "en")[0])

    def test_native_scripts_are_never_replaced_by_guessed_vocables(self):
        for phrase in ("愛你", "あい", "사랑해", "愛愛愛愛愛愛愛愛"):
            with self.subTest(phrase=phrase):
                self.assertEqual(_clean_vocalise_text(phrase, "en")[0], phrase)

    def test_high_compression_vocables_survive_but_runaway_spam_flags(self):
        # Authentic repeated vocables (even 10-40 real beats!) MUST keep their exact count — never deduplicated!
        self.assertFalse(_is_runaway_token_loop("Oh, " * 40, comp_ratio=26.2))
        self.assertEqual(_clean_vocalise_text("Oh, " * 40, "en", comp_ratio=26.2)[0], ("Oh, " * 40).strip())
        self.assertFalse(_is_runaway_token_loop("On wide horizons, " + " ".join(["ah"] * 8), duration_sec=6.0))
        self.assertFalse(_is_runaway_token_loop("La, " * 24, duration_sec=12.0))
        # True decoder runaway degeneration (physically impossible syllable rate or 50-500x flood) IS caught!
        loop_text = "On wide horizons, " + " ".join(["ah"] * 200)
        self.assertTrue(_is_runaway_token_loop(loop_text, duration_sec=24.4))
        self.assertTrue(_is_runaway_token_loop(" ".join(["ah"] * 60), duration_sec=8.0))
        trimmed, did_trim = sanitize_runaway_repetition(loop_text, duration_sec=24.4)
        self.assertTrue(did_trim)
        self.assertEqual(trimmed, "On wide horizons")
        self.assertEqual(sanitize_runaway_repetition(" ".join(["ah"] * 60), duration_sec=8.0), ("", True))
        # A real lyric sentence with trailing vocables is classified as lyric, not pure Vocalise
        self.assertEqual(classify_vocal_event("On wide horizons, ah ah ah ah", fallback_lang="en")[0], "lyric")
        # Chinese/ambiguous bare vocables do not misclassify as lexical lyrics or force a language
        self.assertEqual(classify_vocal_event("啊。", fallback_lang="zh"), ("vocable", True, None))
        self.assertIsNone(_clean_vocalise_text("No, " * 20, "en", comp_ratio=6.5)[0])
        self.assertIsNotNone(HALLUCINATION_PATTERNS.match("Thank you for watching"))

    def test_short_and_quiet_recognized_events_survive(self):
        quiet = np.zeros(120, dtype=np.float32)
        for end in (1.12, 2.0):
            seg = {"start": 1.0, "end": end, "text": "Ah", "text_raw": "Ah", "event_type": "vocable"}
            result = _align_vocalise_to_envelope(seg, quiet, 0.005, 6.0)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["text"], "Ah")
            self.assertEqual(result[0]["text_raw"], "Ah")
            if end > 1.25:
                self.assertEqual(result[0]["review_status"], "needs_review")

    def test_acoustic_bursts_cannot_duplicate_an_untimed_phrase(self):
        rms = np.zeros(240, dtype=np.float32)
        rms[40:100] = 0.08
        rms[160:220] = 0.08
        seg = {"start": 1.0, "end": 12.0, "text": "Oh, ah", "event_type": "vocable", "words": []}
        result = _align_vocalise_to_envelope(seg, rms, 0.004, 12.0)
        self.assertEqual([item["text"] for item in result], ["Oh, ah"])
        self.assertIn("vocal_timing_unresolved", result[0]["review_reasons"])

    def test_timed_syllables_are_partitioned_once_across_bursts(self):
        rms = np.zeros(240, dtype=np.float32)
        rms[40:100] = 0.08
        rms[160:220] = 0.08
        words = [
            {"word": "Oh,", "start": 2.0, "end": 3.0},
            {"word": " ah", "start": 8.0, "end": 9.0},
        ]
        seg = {"start": 1.0, "end": 12.0, "text": "Oh, ah", "event_type": "vocable", "words": words}
        result = _align_vocalise_to_envelope(seg, rms, 0.004, 12.0)
        self.assertEqual([item["text"] for item in result], ["Oh,", "ah"])
        self.assertEqual([word for item in result for word in item["words"]], words)

    def test_every_vocal_occurrence_survives_even_with_identical_coarse_timing(self):
        events = [
            {"start": 1.0, "end": 1.12, "text": "Ah", "event_type": "vocable", "stream_id": "stream_a"},
            {"start": 1.0, "end": 1.12, "text": "Ah", "event_type": "vocable", "stream_id": "stream_a"},
            {"start": 1.13, "end": 1.24, "text": "Ah", "event_type": "vocable", "stream_id": "stream_a"},
            {"start": 1.0, "end": 1.12, "text": "Ah", "event_type": "vocable", "stream_id": "stream_b"},
        ]
        result = _enforce_non_overlapping_same_stream(events)
        self.assertEqual(len(result), len(events))
        self.assertEqual([item["text"] for item in result], ["Ah"] * 4)
        self.assertEqual(
            sorted((item["start"], item["end"]) for item in result),
            sorted((item["start"], item["end"]) for item in events),
        )
        concurrent = [item for item in result if item["start"] == 1.0]
        self.assertEqual(len({item["stream_id"] for item in concurrent}), 3)
        self.assertEqual([item["stream_id"] for item in events], ["stream_a", "stream_a", "stream_a", "stream_b"])

    def test_word_validation_keeps_each_repeated_syllable(self):
        words = [
            {"word": "La", "start": 0.0, "end": 0.2},
            {"word": "la", "start": 0.2, "end": 0.4},
            {"word": "la", "start": 0.4, "end": 0.6},
        ]
        result = _validate_words(words, 0.0, 0.6)
        self.assertEqual([word["word"].strip() for word in result], ["La", "la", "la"])
        self.assertEqual("".join(word["word"] for word in result), "La la la")
        self.assertEqual([(word["start"], word["end"]) for word in result], [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6)])

    def test_repeated_lexical_callouts_keep_adlib_classification(self):
        for phrase in ("yeah yeah yeah", "Yeah, yeah, yeah", " ".join(["hey"] * 12), " ".join(["come on"] * 12)):
            with self.subTest(phrase=phrase):
                event_type, is_adlib, _ = classify_vocal_event(phrase, fallback_lang="en")
                self.assertEqual(event_type, "lyric")
                self.assertTrue(is_adlib)
                cleaned, clean_is_adlib = _clean_vocalise_text(phrase, "en", allow_rhythmic=True)
                self.assertEqual(cleaned, phrase)
                self.assertTrue(clean_is_adlib)
                self.assertFalse(_is_runaway_token_loop(phrase, comp_ratio=8.0))


if __name__ == "__main__":
    unittest.main()
