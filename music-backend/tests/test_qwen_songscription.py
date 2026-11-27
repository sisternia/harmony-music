"""Regression tests for Qwen singing recognition and preservation of overlapping ad-libs."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from cover_studio.music_asr_engine import (
    DEFAULT_MODEL_ID,
    QWEN3_ASR_ID,
    QWEN3_ASR_REPO,
    Qwen3ASREngine,
    SONGSCRIPTION_MODEL_SPECS,
    _enforce_non_overlapping_same_stream,
    _parse_qwen_raw_output,
    create_music_asr_engine,
    normalize_model_id,
    run_music_asr_pipeline,
)


class QwenSongscriptionTests(unittest.TestCase):
    def make_engine(self, text="La la la doo-ba", language="Vietnamese"):
        engine = Qwen3ASREngine.__new__(Qwen3ASREngine)
        engine.repo_id = QWEN3_ASR_REPO
        engine.model_id = QWEN3_ASR_ID
        engine.model = MagicMock()
        engine.model._infer_asr.return_value = [f"language {language}<asr_text>{text}"]
        engine._aligner = None
        engine._alignment_unavailable = False
        engine._log = MagicMock()
        return engine

    def test_registry_defaults_to_qwen_and_upgrades_removed_models(self):
        self.assertEqual(DEFAULT_MODEL_ID, QWEN3_ASR_ID)
        self.assertEqual(normalize_model_id(QWEN3_ASR_ID), QWEN3_ASR_ID)
        self.assertEqual(normalize_model_id("gemini-songscription"), QWEN3_ASR_ID)
        self.assertEqual(normalize_model_id(None), QWEN3_ASR_ID)
        self.assertEqual(normalize_model_id("whisper-large-v3"), "whisper-large-v3")
        self.assertNotIn("gemini-songscription", {s["id"] for s in SONGSCRIPTION_MODEL_SPECS})
        self.assertIn(QWEN3_ASR_ID, {s["id"] for s in SONGSCRIPTION_MODEL_SPECS})
        with patch("cover_studio.music_asr_engine.Qwen3ASREngine") as qwen, patch(
            "cover_studio.music_asr_engine.WhisperLargeV3Engine"
        ) as whisper:
            self.assertIs(create_music_asr_engine(QWEN3_ASR_ID), qwen.return_value)
            whisper.assert_not_called()

    def test_missing_optional_runtime_reports_error_without_substituting_whisper(self):
        with patch.dict(sys.modules, {"qwen_asr": None}), patch(
            "cover_studio.music_asr_engine.WhisperLargeV3Engine"
        ) as whisper:
            with self.assertRaisesRegex(RuntimeError, "qwen-asr==0.0.6"):
                create_music_asr_engine(QWEN3_ASR_ID)
            whisper.assert_not_called()

    def test_transcribe_keeps_qwen_text_and_marks_whisper_alignment_provenance(self):
        engine = self.make_engine()
        audio = np.zeros(16000, dtype=np.float32)
        words = [{"word": "La", "start": 10.0, "end": 10.3, "timing_source": "forced_aligner"}]
        engine._aligner = MagicMock()
        engine._aligner.force_align_chunk.return_value = words
        result = engine.transcribe_chunk(audio, offset_sec=10.0, total_duration=20.0, language_iso="vi")
        self.assertEqual(result["text"], "La la la doo-ba")
        self.assertEqual(result["timing_source"], "forced_aligner")
        self.assertEqual(result["words"], words)
        self.assertEqual(engine.model._infer_asr.call_args.kwargs["languages"], ["Vietnamese"])
        self.assertEqual(engine.model._infer_asr.call_args.kwargs["contexts"], [""])
        engine.model.transcribe.assert_not_called()
        engine._aligner.transcribe_chunk.assert_not_called()
        self.assertEqual(engine._aligner.force_align_chunk.call_args.args[1], result["text"])
        self.assertEqual(engine._aligner.force_align_chunk.call_args.kwargs["offset_sec"], 10.0)

    def test_alignment_failure_retains_text_without_fabricated_word_timestamps(self):
        engine = self.make_engine()
        with patch("cover_studio.music_asr_engine.WhisperLargeV3Engine", side_effect=RuntimeError("unavailable")) as aligner:
            result = engine.transcribe_chunk(np.zeros(16000), offset_sec=2.0, total_duration=10.0)
            second = engine.transcribe_chunk(np.zeros(16000), offset_sec=4.0, total_duration=10.0)
        self.assertEqual(result["text"], "La la la doo-ba")
        self.assertEqual(result["words"], [])
        self.assertEqual(result["timing_source"], "asr_segment")
        self.assertEqual(second["text"], result["text"])
        self.assertEqual(aligner.call_count, 1)

    def test_auto_language_retains_mixed_language_output_and_skips_aligner_when_unneeded(self):
        engine = self.make_engine(language="Japanese,English")
        result = engine.transcribe_chunk(
            np.zeros(16000), offset_sec=0.0, total_duration=1.0, want_timestamps=False,
        )
        self.assertEqual(engine.model._infer_asr.call_args.kwargs["languages"], [None])
        self.assertEqual(result["language"], "ja")
        self.assertEqual(result["language_raw"], "Japanese,English")
        self.assertIsNone(engine._aligner)

    def test_raw_parser_and_inference_keep_long_repeated_vocables_and_scat(self):
        for phrase in ("la", "doo-ba-da"):
            text = " ".join([phrase] * 24)
            self.assertEqual(_parse_qwen_raw_output(f"language English<asr_text>{text}"), ("English", text))
            self.assertEqual(_parse_qwen_raw_output(text, "English"), ("English", text))
            engine = self.make_engine(text=text, language="English")
            result = engine.transcribe_chunk(
                np.zeros(16000), offset_sec=0.0, total_duration=1.0, want_timestamps=False,
            )
            self.assertEqual(result["text"], text)
            engine.model.transcribe.assert_not_called()

    def test_incompatible_raw_runtime_fails_without_using_destructive_sdk_parser(self):
        engine = self.make_engine()
        engine.model = types.SimpleNamespace(transcribe=MagicMock())
        with self.assertRaisesRegex(RuntimeError, "qwen-asr==0.0.6"):
            engine.transcribe_chunk(np.zeros(16000), offset_sec=0.0, total_duration=1.0)
        engine.model.transcribe.assert_not_called()

    def test_contained_adlib_preserves_both_performances_and_original_word_times(self):
        words = [{"word": "Lead", "start": 1.0, "end": 4.0}]
        lead = {"start": 1.0, "end": 4.0, "text": "Lead", "words": words}
        adlib = {"start": 2.0, "end": 3.0, "text": "Ooh", "event_type": "vocable", "is_adlib": True}
        later = {**adlib, "start": 4.0, "end": 5.0}
        result = _enforce_non_overlapping_same_stream([lead, adlib, later, dict(adlib)])
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["words"], words)
        self.assertEqual((result[1]["start"], result[1]["end"]), (2.0, 3.0))
        self.assertEqual((result[2]["start"], result[2]["end"]), (2.0, 3.0))
        self.assertNotEqual(result[1]["stream_id"], result[2]["stream_id"])
        self.assertNotEqual(result[0].get("stream_id", "stream_a"), result[1]["stream_id"])
        self.assertEqual([item["text"] for item in result], ["Lead", "Ooh", "Ooh", "Ooh"])
        self.assertTrue(result[0]["overlap_group_id"])
        self.assertEqual(result[0]["overlap_group_id"], result[1]["overlap_group_id"])
        self.assertEqual(lead["words"], words)

    @patch("cover_studio.vocal_recovery.recover_vocals_with_asr", return_value=([], {"status": "ok"}))
    def test_pipeline_dispatch_and_forced_alignment_provenance(self, _recover):
        engine = self.make_engine(text="Hello there", language="English")
        engine._aligner = MagicMock()
        engine._aligner.force_align_chunk.return_value = [
            {"word": "Hello", "start": 0.0, "end": 0.45, "timing_source": "forced_aligner"},
            {"word": " there", "start": 0.45, "end": 1.0, "timing_source": "forced_aligner"},
        ]
        audio = np.ones(16000, dtype=np.float32) * 0.1
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine) as factory:
            result = run_music_asr_pipeline(
                asr_audio=audio, duration=1.0, vocal_isolated=False,
                rms_env=np.ones(20) * 0.1, vocal_floor=0.005,
                requested_lang="en", model_id=QWEN3_ASR_ID,
                progress=lambda *_: None, should_stop=lambda: False, log=lambda _: None,
            )
        factory.assert_called_once_with(QWEN3_ASR_ID, log=unittest.mock.ANY)
        self.assertEqual(result["diagnostics"]["config"]["asr_repo"], QWEN3_ASR_REPO)
        self.assertEqual(result["diagnostics"]["recognition_engine"], QWEN3_ASR_REPO)
        self.assertEqual(result["diagnostics"]["word_alignment"], "whisper-large-v3-dtw")
        self.assertEqual(result["raw_segments"][0]["timing_source"], "forced_aligner")
        self.assertEqual(result["raw_segments"][0]["text"], "Hello there")

    @patch("cover_studio.vocal_recovery.recover_vocals_with_asr", return_value=([], {"status": "ok"}))
    def test_reference_replacement_keeps_audio_recognized_vocables(self, _recover):
        engine = self.make_engine(text="Ooh", language="English")
        engine._aligner = MagicMock()
        engine._aligner.force_align_chunk.return_value = [
            {"word": "Ooh", "start": 0.2, "end": 0.8, "timing_source": "forced_aligner"},
        ]
        reference = {
            "source": "test", "track_name": "Song", "artist_name": "Artist",
            "match_score": 1.0, "time_offset": 0.0,
            "synced_lines": [{"start": 0.0, "text": "Main lyric"}] * 4,
        }
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine), patch(
            "cover_studio.lyrics_reference.find_verified_reference_lyrics", return_value=reference,
        ), patch(
            "cover_studio.music_asr_engine.align_reference_lines_with_whisper",
            return_value=[{"start": 0.0, "end": 1.0, "text": "Main lyric", "words": []}],
        ):
            result = run_music_asr_pipeline(
                asr_audio=np.ones(16000, dtype=np.float32) * 0.1, duration=1.0,
                vocal_isolated=False, rms_env=np.ones(20) * 0.1, vocal_floor=0.005,
                requested_lang="en", model_id=QWEN3_ASR_ID, track_title="Song",
                progress=lambda *_: None, should_stop=lambda: False, log=lambda _: None,
            )
        self.assertEqual({s["text"] for s in result["raw_segments"]}, {"Ooh", "Main lyric"})
        self.assertEqual(result["raw_segments"][0]["start"], 0.0)
        self.assertNotEqual(
            result["raw_segments"][0].get("stream_id", "stream_a"),
            result["raw_segments"][1]["stream_id"],
        )

    @patch("cover_studio.vocal_recovery.recover_vocals_with_asr", return_value=([], {"status": "ok"}))
    def test_quiet_recognized_line_is_flagged_instead_of_deleted(self, _recover):
        engine = self.make_engine(text="Hello there", language="English")
        engine._aligner = MagicMock()
        engine._aligner.force_align_chunk.return_value = [
            {"word": "Hello", "start": 0.0, "end": 0.4, "timing_source": "forced_aligner"},
            {"word": " there", "start": 0.4, "end": 0.9, "timing_source": "forced_aligner"},
        ]
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine), patch(
            "cover_studio.lyrics_service._extract_vocal_regions", return_value=[(0.0, 1.0)],
        ):
            result = run_music_asr_pipeline(
                asr_audio=np.ones(16000, dtype=np.float32) * 0.001, duration=1.0,
                vocal_isolated=True, rms_env=np.zeros(20), vocal_floor=0.005,
                requested_lang="en", model_id=QWEN3_ASR_ID,
                progress=lambda *_: None, should_stop=lambda: False, log=lambda _: None,
            )
        self.assertEqual(result["raw_segments"][0]["text"], "Hello there")
        self.assertEqual(result["raw_segments"][0]["review_status"], "needs_review")

    def test_recovery_runs_once_on_loaded_engine_and_receives_both_audio_sources(self):
        engine = self.make_engine(text="Main lyric", language="English")
        engine.close = MagicMock(wraps=engine.close)
        engine.force_align_chunk = MagicMock(return_value=[])
        vocal = np.ones(16000, dtype=np.float32) * 0.05
        mix = np.ones(16000, dtype=np.float32) * 0.1
        progress_events = []

        def recover(**kwargs):
            self.assertIs(kwargs["asr"], engine)
            self.assertIsNotNone(engine.model)
            engine.close.assert_not_called()
            self.assertIs(kwargs["full_vocal_16k"], vocal)
            self.assertIs(kwargs["original_audio"], mix)
            return ([{
                "start": 0.2, "end": 0.8, "text": "Ooh", "event_type": "vocable",
                "is_adlib": True, "vocal_role": "backing", "words": [],
            }], {"status": "ok", "recovered_count": 1})

        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine) as factory, patch(
            "cover_studio.vocal_recovery.recover_vocals_with_asr", side_effect=recover,
        ) as recovery:
            result = run_music_asr_pipeline(
                asr_audio=mix, pure_vocal_audio=vocal, original_audio=mix,
                duration=1.0, vocal_isolated=True, rms_env=np.ones(20) * 0.05, vocal_floor=0.005,
                requested_lang="en", model_id=QWEN3_ASR_ID,
                progress=lambda p, stage: progress_events.append((p, stage)),
                should_stop=lambda: False, log=lambda _: None,
            )
        factory.assert_called_once()
        recovery.assert_called_once()
        engine.close.assert_called_once()
        self.assertEqual(result["diagnostics"]["adlib_recovery"]["recovered_count"], 1)
        self.assertEqual({s["text"] for s in result["raw_segments"]}, {"Main lyric", "Ooh"})
        self.assertIn((0.78, "reviewing_vocals"), progress_events)
        self.assertEqual([p for p, _ in progress_events], sorted(p for p, _ in progress_events))

    @patch("cover_studio.vocal_recovery.recover_vocals_with_asr", return_value=([], {"status": "ok"}))
    def test_runaway_vocalise_loop_splits_chunk_and_recovers_real_lyrics(self, _recover):
        engine = self.make_engine(text="unused", language="English")
        engine.force_align_chunk = MagicMock(return_value=[])
        # First call on full 20s chunk hits Qwen runaway "ah ah ah..." loop; split sub-chunks recover cleanly
        loop_raw = "language English<asr_text>On wide horizons, " + " ".join(["ah"] * 180)
        engine.model._infer_asr.side_effect = [
            [loop_raw],
            ["language English<asr_text>On wide horizons we fly."],
            ["language English<asr_text>Chasing the dawn ahead."],
            ["language English<asr_text>Waiting for tomorrow."],
        ]
        audio = np.ones(16000 * 20, dtype=np.float32) * 0.1
        rms_env = np.ones(400, dtype=np.float32) * 0.1
        rms_env[130:140] = 0.004  # valley near 6.5s
        rms_env[265:275] = 0.004  # valley near 13.5s
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine), patch(
            "cover_studio.lyrics_service._extract_vocal_regions", return_value=[(0.0, 20.0)],
        ):
            result = run_music_asr_pipeline(
                asr_audio=audio, duration=20.0, vocal_isolated=True,
                rms_env=rms_env, vocal_floor=0.005,
                requested_lang="en", model_id=QWEN3_ASR_ID,
                progress=lambda *_: None, should_stop=lambda: False, log=lambda _: None,
            )
        texts = [s["text"] for s in result["raw_segments"]]
        self.assertEqual(texts, ["On wide horizons we fly.", "Chasing the dawn ahead.", "Waiting for tomorrow."])
        self.assertTrue(all("ah ah ah" not in t for t in texts))

    @patch("cover_studio.vocal_recovery.recover_vocals_with_asr", return_value=([], {"status": "ok"}))
    def test_short_chunk_runaway_tail_trims_cleanly_instead_of_spamming(self, _recover):
        engine = self.make_engine(text="On wide horizons, " + " ".join(["ah"] * 90), language="English")
        engine.force_align_chunk = MagicMock(return_value=[])
        audio = np.ones(16000 * 5, dtype=np.float32) * 0.1
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine), patch(
            "cover_studio.lyrics_service._extract_vocal_regions", return_value=[(0.0, 5.0)],
        ):
            result = run_music_asr_pipeline(
                asr_audio=audio, duration=5.0, vocal_isolated=True,
                rms_env=np.ones(100, dtype=np.float32) * 0.1, vocal_floor=0.005,
                requested_lang="en", model_id=QWEN3_ASR_ID,
                progress=lambda *_: None, should_stop=lambda: False, log=lambda _: None,
            )
        self.assertEqual(len(result["raw_segments"]), 1)
        seg = result["raw_segments"][0]
        self.assertEqual(seg["text"], "On wide horizons")
        self.assertEqual(seg["event_type"], "lyric")
        self.assertEqual(seg["review_status"], "needs_review")
        self.assertIn("asr_repetition_loop_trimmed", seg["review_reasons"])

    @patch("cover_studio.vocal_recovery.recover_vocals_with_asr")
    def test_alignment_mode_does_not_add_recovery_vocals_to_user_lyrics(self, recover):
        engine = self.make_engine(text="Main lyric", language="English")
        engine.force_align_chunk = MagicMock(return_value=[])
        with patch("cover_studio.music_asr_engine.create_music_asr_engine", return_value=engine):
            run_music_asr_pipeline(
                asr_audio=np.ones(16000, dtype=np.float32) * 0.1,
                duration=1.0, vocal_isolated=False, rms_env=np.ones(20) * 0.1, vocal_floor=0.005,
                requested_lang="en", model_id=QWEN3_ASR_ID, mode="align",
                progress=lambda *_: None, should_stop=lambda: False, log=lambda _: None,
            )
        recover.assert_not_called()


if __name__ == "__main__":
    unittest.main()
