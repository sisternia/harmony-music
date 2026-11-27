"""Partial alignment must never replace recognized text with fewer sung syllables."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from cover_studio.music_asr_engine import (
    QWEN3_ASR_ID,
    QWEN3_ASR_REPO,
    _group_words_into_lyric_lines,
    _refine_line_with_second_pass_aligner,
    _vocalise_adds_occurrences,
    _vocalise_loses_occurrences,
    align_reference_lines_with_whisper,
    refine_segments_with_music_asr,
    run_music_asr_pipeline,
)


class AlignmentTextPreservationTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.full(64_000, .1, dtype=np.float32)
        self.rms = np.full(80, .1, dtype=np.float32)
        self.partial = [{"word": "Ooh", "start": 1., "end": 2., "timing_source": "forced_aligner"}]

    def make_engine(self, text="Ooh la la", words=None):
        engine = MagicMock()
        engine.repo_id = QWEN3_ASR_REPO
        engine.transcribe_chunk.return_value = {
            "text": text, "language": "en", "words": self.partial if words is None else words,
            "timing_source": "forced_aligner",
        }
        engine.force_align_chunk.return_value = self.partial
        return engine

    def test_line_grouping_retains_unaligned_syllables_with_coarse_bounds(self):
        result = _group_words_into_lyric_lines(self.partial, "Ooh la la", 1., 4., is_vocalise_chunk=True)
        self.assertEqual(result, [("Ooh la la", 1., 4., [])])

    def test_line_grouping_does_not_treat_repeated_syllables_as_a_set(self):
        words = [{"word": "La", "start": 1., "end": 2.}, {"word": " la", "start": 2., "end": 3.}]
        self.assertEqual(
            _group_words_into_lyric_lines(words, "La la la", 1., 4., is_vocalise_chunk=True),
            [("La la la", 1., 4., [])],
        )

    def test_pipeline_keeps_full_recognized_and_raw_text_with_partial_alignment(self):
        engine = self.make_engine()
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine), \
             patch("cover_studio.vocal_recovery.recover_vocals_with_asr", return_value=([], {"status": "ok"})):
            result = run_music_asr_pipeline(
                asr_audio=self.audio, duration=4., vocal_isolated=False,
                rms_env=self.rms, vocal_floor=.005, requested_lang="en", model_id=QWEN3_ASR_ID,
                progress=lambda *_: None, should_stop=lambda: False, log=lambda _: None,
            )
        self.assertEqual([s["text"] for s in result["raw_segments"]], ["Ooh la la"])
        self.assertEqual(result["raw_segments"][0]["text_raw"], "Ooh la la")
        self.assertEqual(result["raw_segments"][0]["words"], [])
        self.assertEqual(result["anchors"], [])
        self.assertEqual(result["raw_asr_segments"][0]["text"], "Ooh la la")
        engine.close.assert_called_once()

    def test_second_alignment_cannot_shorten_three_repetitions_to_two(self):
        original = [
            {"word": "Oh", "start": 1., "end": 1.5},
            {"word": " oh", "start": 1.5, "end": 2.},
            {"word": " oh", "start": 2., "end": 3.},
        ]
        engine = self.make_engine()
        engine.force_align_chunk.return_value = [dict(w) for w in original[:2]]
        result = _refine_line_with_second_pass_aligner(
            asr=engine, asr_audio=self.audio, line_text="Oh oh oh", ls=1., le=3.,
            stage1_words=original, raw_words_had_collapse=True, reg_s=0., reg_e=4., duration=4.,
            lang_iso="en", rms_env=self.rms, vocal_floor=.005,
        )
        self.assertEqual(result, (1., 3., original))
        engine.force_align_chunk.assert_called_once()

    def test_vocalise_occurrence_guard_distinguishes_loss_from_acoustic_recovery(self):
        self.assertTrue(_vocalise_loses_occurrences("Ah ah", "Ah"))
        self.assertTrue(_vocalise_adds_occurrences("Ah", "Ah ah"))
        self.assertFalse(_vocalise_loses_occurrences("Ah", "Ah ah"))

    def test_reread_prefers_candidate_that_recovers_repeated_vocalise_syllable(self):
        engine = self.make_engine()
        engine.transcribe_chunk.side_effect = [
            {"text": "Ah", "language": "en", "words": []},
            {"text": "Ah ah", "language": "en", "words": []},
        ]
        original = {
            "id": 1, "start": 1., "end": 2., "text": "Ah", "text_raw": "Ah",
            "event_type": "vocable", "is_adlib": True,
        }
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine):
            updated, reports = refine_segments_with_music_asr(
                segments=[original], target_indices=[0], pure_vocal=self.audio, crisp_blend=self.audio,
                duration=4., has_vocal_stem=True, decode_lang="en", model_id=QWEN3_ASR_ID,
            )
        self.assertEqual(updated[0]["text"], "Ah ah")
        self.assertEqual(updated[0]["text_raw"], "Ah")
        self.assertEqual(reports[0]["new_text"], "Ah ah")
        self.assertEqual(original["text"], "Ah")

    def test_reread_does_not_overwrite_two_vocalise_syllables_with_one(self):
        engine = self.make_engine()
        engine.transcribe_chunk.side_effect = [
            {"text": "Ah", "language": "en", "words": []},
            {"text": "Ah", "language": "en", "words": []},
        ]
        original = {
            "id": 1, "start": 1., "end": 2., "text": "Ah ah", "text_raw": "Ah ah",
            "event_type": "vocable", "is_adlib": True,
        }
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine):
            updated, reports = refine_segments_with_music_asr(
                segments=[original], target_indices=[0], pure_vocal=self.audio, crisp_blend=self.audio,
                duration=4., has_vocal_stem=True, decode_lang="en", model_id=QWEN3_ASR_ID,
            )
        self.assertEqual(updated[0]["text"], "Ah ah")
        self.assertEqual(reports[0]["new_text"], "Ah ah")
        self.assertFalse(reports[0]["changed"])

    def test_reference_partial_alignment_keeps_full_text_and_estimated_end(self):
        engine = self.make_engine()
        result = align_reference_lines_with_whisper(
            asr=engine, asr_audio=self.audio, duration=4.,
            lines=[{"text": "Ooh la la", "start": 1., "end": 4.}], primary_lang="en",
            rms_env=self.rms, vocal_floor=.005, source_variant="vocal_stem",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Ooh la la")
        self.assertEqual(result[0]["text_raw"], "Ooh la la")
        self.assertEqual((result[0]["start"], result[0]["end"]), (1., 4.))
        self.assertEqual(result[0]["words"], [])
        self.assertEqual(result[0]["review_status"], "needs_review")
        self.assertIn("vocal_timing_unresolved", result[0]["review_reasons"])

    def test_refinement_retains_full_raw_recognition_without_partial_word_timing(self):
        engine = self.make_engine()
        original = {"id": 1, "start": 1., "end": 4., "text": "Ooh", "text_raw": "Ooh"}
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine):
            updated, reports = refine_segments_with_music_asr(
                segments=[original], target_indices=[0], pure_vocal=self.audio, crisp_blend=self.audio,
                duration=4., has_vocal_stem=True, decode_lang="en", model_id=QWEN3_ASR_ID,
            )
        self.assertEqual(updated[0]["text"], "Ooh la la")
        self.assertEqual(updated[0]["words"], [])
        self.assertEqual(updated[0]["text_raw"], "Ooh")
        self.assertEqual(updated[0]["timing_granularity"], "line")
        self.assertEqual(updated[0]["source_audio_variant"], "vocal_stem")
        self.assertEqual(reports[0]["new_text"], "Ooh la la")
        self.assertIn("Kênh vocal", reports[0]["variant"])
        self.assertTrue(all(c["text"] == "Ooh la la" for c in updated[0]["candidates"]))
        self.assertEqual(original["text"], "Ooh")
        engine.close.assert_called_once()

    def test_reread_prioritizes_pure_vocal_stem_over_mix_and_avoids_duplicate_mix_pass(self):
        pure = np.full(64_000, 0.25, dtype=np.float32)
        blended = np.full(64_000, 0.50, dtype=np.float32)
        seen_means: list[float] = []

        def fake_transcribe(arr, **_kw):
            seen_means.append(round(float(np.mean(arr)), 2))
            if float(np.mean(arr)) < 0.35:
                return {"text": "Clear vocal lyric", "language": "en", "words": []}
            return {"text": "Noisy mix lyric", "language": "en", "words": []}

        engine = self.make_engine()
        engine.transcribe_chunk.side_effect = fake_transcribe
        original = {"id": 1, "start": 1., "end": 3., "text": "Old lyric", "text_raw": "Old lyric"}
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine):
            updated, reports = refine_segments_with_music_asr(
                segments=[original], target_indices=[0], pure_vocal=pure, crisp_blend=blended,
                duration=4., has_vocal_stem=True, decode_lang="en", model_id=QWEN3_ASR_ID,
            )
        self.assertEqual(seen_means, [0.25, 0.50])
        self.assertEqual(updated[0]["text"], "Clear vocal lyric")
        self.assertEqual(updated[0]["source_audio_variant"], "vocal_stem")
        self.assertIn("Kênh vocal", reports[0]["variant"])

        # Without a vocal stem, never run the identical mix twice and falsely auto-resolve
        engine_mix = self.make_engine(text="Only mix lyric", words=[])
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine_mix):
            upd_mix, rep_mix = refine_segments_with_music_asr(
                segments=[dict(original)], target_indices=[0], pure_vocal=blended, crisp_blend=blended,
                duration=4., has_vocal_stem=False, decode_lang="en", model_id=QWEN3_ASR_ID,
            )
        self.assertEqual(engine_mix.transcribe_chunk.call_count, 1)
        self.assertEqual(upd_mix[0]["source_audio_variant"], "original_mix")
        self.assertFalse(rep_mix[0]["auto_resolved"])


if __name__ == "__main__":
    unittest.main()
