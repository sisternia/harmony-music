"""Regression tests for repeated, missing and overlapping sung syllables."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from cover_studio.engine import Cancelled
from cover_studio.lyrics_service import compute_vocal_envelope
from cover_studio.vocal_recovery import (
    detect_vocalise_syllable_pulses,
    find_adlib_review_windows,
    find_uncovered_vocal_gaps,
    merge_recovered_vocal_events,
    recover_vocals_with_asr,
)


class VocalRecoveryTests(unittest.TestCase):
    @staticmethod
    def audio(seconds=4):
        t = np.arange(int(seconds * 16000)) / 16000
        return (.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def run_recovery(self, asr, segments=None, **kw):
        audio = self.audio()
        rms, floor = compute_vocal_envelope(audio)
        return recover_vocals_with_asr(
            asr=asr, segments=segments or [], full_vocal_16k=audio,
            rms_env=rms, vocal_floor=floor, duration=4., primary_lang="en",
            vocal_isolated=True, **kw,
        )

    @staticmethod
    def window(start=1., end=2., occupied=False):
        return dict(start=start, end=end, crop_start=max(0, start - .2),
                    crop_end=end + .2, occupied=occupied)

    def test_lanes_preserve_every_repeated_occurrence_and_original_input(self):
        lead = dict(start=0., end=5., text="Main lyric", stream_id="stream_a")
        events = [dict(start=s, end=e, text=text, stream_id="stream_b") for s, e, text in
                  [(1., 2., "Ooh"), (1.05, 2.05, "Ooh"), (1.1, 1.8, "Hey"), (3., 4., "Ooh")]]
        result = merge_recovered_vocal_events([lead], events)
        self.assertEqual([s["text"] for s in result], ["Main lyric", "Ooh", "Ooh", "Hey", "Ooh"])
        self.assertEqual([s["stream_id"] for s in result], ["stream_a", "stream_b", "stream_c", "stream_d", "stream_b"])
        self.assertEqual(result[0], lead)
        self.assertEqual(events[2]["stream_id"], "stream_b")

    def test_exact_observations_match_across_passes_only_one_to_one(self):
        event = dict(start=1., end=2., text="Doo-ba-da", stream_id="stream_a")
        originals = [dict(event) for _ in range(3)]
        recovered = [dict(event) for _ in range(3)]
        self.assertEqual(merge_recovered_vocal_events(originals, recovered), originals)
        result = merge_recovered_vocal_events([dict(event)], recovered)
        self.assertEqual([s["text"] for s in result], ["Doo-ba-da"] * 3)

    def test_backing_may_repeat_lead_words_at_distinct_times(self):
        lead = dict(start=0., end=4., text="Sing oh yeah", vocal_role="lead")
        backing = dict(start=1., end=2., text="oh yeah", vocal_role="unknown")
        repeated = {**backing, "start": 1.03, "end": 2.03}
        result = merge_recovered_vocal_events([lead], [backing, repeated])
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1]["text"], "oh yeah")
        self.assertEqual(result[1]["stream_id"], "stream_b")
        short = {**backing, "start": 1.2, "end": 1.5, "text": "yeah"}
        self.assertEqual(len(merge_recovered_vocal_events([backing], [short])), 2)

    def test_unknown_placeholder_is_retired_only_when_entire_region_resolved(self):
        unknown = dict(start=1., end=2., text="", event_type="unknown_vocal")
        phrase = dict(start=1., end=2., text="Ooh")
        self.assertEqual([s["text"] for s in merge_recovered_vocal_events([unknown], [phrase])], ["Ooh"])
        self.assertEqual(len(merge_recovered_vocal_events([unknown], [{**phrase, "end": 1.4}])), 2)

    def test_gap_scan_has_no_fourteen_phrase_limit(self):
        audio = np.zeros(16_000 * 32, dtype=np.float32)
        for i in range(16):
            audio[i * 32000:i * 32000 + 6400] = self.audio(.4)
        rms, floor = compute_vocal_envelope(audio)
        gaps = find_uncovered_vocal_gaps(
            segments=[], full_vocal_16k=audio, rms_env=rms, vocal_floor=floor, duration=32.,
        )
        self.assertEqual(len(gaps), 16)


    def test_word_level_mask_finds_missing_syllables_inside_lyric(self):
        audio = self.audio()
        rms = np.full(80, .1)
        lead = dict(start=0., end=4., text="Main lyric", words=[
            dict(word="Main", start=0., end=1.), dict(word=" lyric", start=2., end=4.),
        ])
        gaps = find_uncovered_vocal_gaps(
            segments=[lead], full_vocal_16k=audio, rms_env=rms, vocal_floor=.005, duration=4.,
        )
        self.assertEqual(len(gaps), 1)
        self.assertGreaterEqual(gaps[0]["start"], 1.)
        self.assertLessEqual(gaps[0]["end"], 2.)

    def test_clear_lyrics_are_not_reread_and_suspect_windows_do_not_overlap(self):
        audio = self.audio()
        kwargs = dict(full_vocal_16k=audio, rms_env=np.full(80, .1), vocal_floor=.005, duration=4.)
        lead = dict(start=0., end=4., text="Main lyric")
        self.assertEqual(find_adlib_review_windows(segments=[lead], **kwargs), [])
        suspect = {**lead, "low_confidence": True}
        windows = find_adlib_review_windows(segments=[suspect, dict(suspect)], **kwargs)
        self.assertEqual(len(windows), 1)
        self.assertTrue(windows[0]["occupied"])
        verified = {**suspect, "review_status": "verified"}
        self.assertEqual(find_adlib_review_windows(segments=[verified], **kwargs), [])

    def test_single_vocable_gets_an_occupied_retry_window(self):
        audio = self.audio()
        kwargs = dict(full_vocal_16k=audio, rms_env=np.full(80, .1), vocal_floor=.005, duration=4.)
        single = dict(start=1., end=2., text="Ah", event_type="vocable", is_adlib=True)
        multi = dict(start=2., end=3., text="Ah ah", event_type="vocable", is_adlib=True)
        windows = find_adlib_review_windows(segments=[single, multi], **kwargs)
        occupied = [window for window in windows if window["occupied"]]
        self.assertEqual(len(occupied), 1)
        self.assertEqual((occupied[0]["start"], occupied[0]["end"]), (1., 2.))

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_retry_keeps_every_scat_syllable_and_applies_offset_once(self, windows):
        windows.return_value = [self.window()]
        words = [dict(word=w, start=s, end=e, timing_source="forced_aligner") for w, s, e in
                 [("Doo", 1.1, 1.3), (" ba", 1.3, 1.5), (" da", 1.5, 1.7), (" da", 1.7, 1.9)]]
        asr = MagicMock()
        asr.transcribe_chunk.return_value = dict(text="Doo ba da da", words=words, language="en", timing_source="forced_aligner")
        result, stats = self.run_recovery(asr)
        self.assertEqual([s["text"] for s in result], ["Doo ba da da"])
        self.assertEqual((result[0]["start"], result[0]["end"]), (1.1, 1.9))
        self.assertEqual(result[0]["vocal_role"], "unknown")
        self.assertEqual(result[0]["review_status"], "needs_review")
        self.assertEqual(stats["recovered_count"], 1)
        self.assertEqual(asr.transcribe_chunk.call_args.kwargs["offset_sec"], 1.)
        asr.close.assert_not_called()

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_occupied_retry_subtracts_lead_but_preserves_inserted_repetition(self, windows):
        windows.return_value = [self.window(0., 4., occupied=True)]
        lead = dict(start=0., end=4., text="Sing yeah", words=[
            dict(word="Sing", start=.2, end=.6), dict(word=" yeah", start=1., end=1.4),
        ])
        words = lead["words"] + [dict(word=" yeah", start=2., end=2.3), dict(word=" yeah", start=2.5, end=2.9)]
        asr = MagicMock()
        asr.transcribe_chunk.return_value = dict(text="Sing yeah yeah yeah", words=words, language="en")
        result, _ = self.run_recovery(asr, [lead])
        self.assertEqual([s["text"] for s in result], ["yeah yeah"])
        self.assertEqual([w["start"] for w in result[0]["words"]], [2., 2.5])

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_occupied_vocalise_retry_recovers_second_ah_occurrence(self, windows):
        windows.return_value = [self.window(1., 2., occupied=True)]
        original = dict(
            start=1., end=2., text="Ah", event_type="vocable", is_adlib=True,
            words=[dict(word="Ah", start=1.2, end=1.45)],
        )
        asr = MagicMock()
        asr.transcribe_chunk.return_value = dict(
            text="Ah ah", language="en", timing_source="forced_aligner",
            words=[dict(word="Ah", start=1.2, end=1.45), dict(word=" ah", start=1.6, end=1.85)],
        )
        result, _ = self.run_recovery(asr, [original])
        self.assertEqual([segment["text"].strip().casefold() for segment in result], ["ah"])
        self.assertEqual(result[0]["start"], 1.6)
        self.assertEqual(result[0]["end"], 1.85)

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_original_mix_retry_only_after_empty_stem_and_untimed_scat_kept(self, windows):
        windows.return_value = [self.window()]
        asr = MagicMock()
        asr.transcribe_chunk.side_effect = [dict(text="", words=[]), dict(text="La la la", words=[], language="en")]
        result, stats = self.run_recovery(asr, original_audio=self.audio())
        self.assertEqual(result[0]["text"], "La la la")
        self.assertEqual(result[0]["source_audio_variant"], "original_mix")
        self.assertEqual(stats["recognition_attempts"], 2)

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_occupied_retry_requires_timing_evidence(self, windows):
        windows.return_value = [self.window(0., 4., occupied=True)]
        asr = MagicMock()
        asr.transcribe_chunk.return_value = dict(text="Main lyric ooh", words=[])
        lead = dict(start=0., end=4., text="Main lyric")
        result, _ = self.run_recovery(asr, [lead])
        self.assertEqual(result, [])
        self.assertEqual(lead["candidates"][0]["text"], "Main lyric ooh")

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_later_existing_repetition_is_matched_by_its_own_time(self, windows):
        windows.return_value = [self.window(0., 4., occupied=True)]
        lead = dict(start=0., end=4., text="Oh Oh", words=[
            dict(word="Oh", start=.2, end=.5), dict(word=" Oh", start=2.2, end=2.5),
        ])
        asr = MagicMock()
        asr.transcribe_chunk.return_value = dict(text="Oh", words=[lead["words"][1]])
        result, _ = self.run_recovery(asr, [lead])
        self.assertEqual(result, [])

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_partial_multiword_token_preserves_added_syllable_with_coarse_time(self, windows):
        windows.return_value = [self.window(0., 4., occupied=True)]
        lead = dict(start=0., end=4., text="Hello", words=[dict(word="Hello", start=1., end=2.)])
        asr = MagicMock()
        asr.transcribe_chunk.return_value = dict(text="Hello yeah", words=[dict(word="Hello yeah", start=1., end=3.)])
        result, _ = self.run_recovery(asr, [lead])
        self.assertEqual(result[0]["text"], "yeah")
        self.assertEqual(result[0]["words"], [])
        self.assertEqual(result[0]["timing_source"], "asr_segment")

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_partial_alignment_keeps_full_scat_text(self, windows):
        windows.return_value = [self.window(1., 3.)]
        asr = MagicMock()
        asr.transcribe_chunk.return_value = dict(text="Ooh la la", words=[dict(word="Ooh", start=1.1, end=1.5)])
        result, _ = self.run_recovery(asr)
        self.assertEqual(result[0]["text"], "Ooh la la")
        self.assertEqual(result[0]["words"], [])

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_invalid_word_times_and_hallucinated_subtitles_are_not_inserted(self, windows):
        windows.return_value = [self.window()]
        asr = MagicMock()
        asr.transcribe_chunk.return_value = dict(text="Hey", words=[
            dict(word="Hey", start=float("nan"), end=1.4),
            dict(word="Hey", start=2.1, end=2.5),
        ])
        result, _ = self.run_recovery(asr)
        self.assertEqual(result[0]["text"], "")
        asr.transcribe_chunk.return_value = dict(text="Thank you for watching", words=[])
        result, _ = self.run_recovery(asr)
        self.assertEqual(result[0]["text"], "")

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_short_unrecognized_vocal_is_flagged_without_invented_syllables(self, windows):
        windows.return_value = [self.window(1., 1.4)]
        asr = MagicMock()
        asr.transcribe_chunk.return_value = dict(text="", words=[])
        result, stats = self.run_recovery(asr)
        self.assertEqual(result[0]["event_type"], "unknown_vocal")
        self.assertEqual(result[0]["text"], "")
        self.assertIsNone(result[0]["confidence"])
        self.assertIsNone(result[0]["language"])
        self.assertEqual(stats["unknown_count"], 1)
        self.assertEqual(len(merge_recovered_vocal_events([], result)), 1)

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_failed_crop_does_not_drop_later_recoveries(self, windows):
        windows.return_value = [self.window(0., 1.), self.window(2., 3.)]
        asr = MagicMock()
        asr.transcribe_chunk.side_effect = [RuntimeError("model error"), dict(text="Hey hey", words=[])]
        result, stats = self.run_recovery(asr)
        self.assertEqual([s["text"] for s in result], ["", "Hey hey"])
        self.assertEqual(stats["status"], "partial")
        self.assertEqual(stats["failed_attempts"], 1)

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_recovery_rejects_runaway_spam_while_preserving_every_real_vocalise(self, windows):
        windows.return_value = [
            {**self.window(1.0, 3.0), "mean_rms": 0.08},
            {**self.window(3.2, 3.55), "mean_rms": 0.025},
            {**self.window(3.6, 3.95), "mean_rms": 0.025},
        ]
        asr = MagicMock()
        asr.transcribe_chunk.side_effect = [
            dict(text=" ".join(["ah"] * 80), words=[], language="en"),
            dict(text="Ah.", words=[dict(word="Ah.", start=3.2, end=3.5)], language="en"),
            dict(text="Ah.", words=[dict(word="Ah.", start=3.6, end=3.9)], language="en"),
        ]
        result, _ = self.run_recovery(asr)
        # First window had impossible runaway spam (80 "ah" in 2s) -> rejected cleanly
        self.assertTrue(all("ah ah ah" not in str(s.get("text") or "").lower() for s in result))
        # Both genuine short "Ah." vocalises keep their EXACT count (2, never deduplicated or dropped!)
        self.assertEqual([s["text"] for s in result if s.get("text") == "Ah."], ["Ah.", "Ah."])

    def test_pulse_detector_separates_ah_ah_without_splitting_sustained_ooh(self):
        sr = 16000
        # 1. Single sustained note (2.0s at 440Hz with gentle 6Hz vibrato) -> MUST stay 1 pulse
        t_long = np.arange(sr * 2, dtype=np.float32) / sr
        vibrato = 1.0 + 0.12 * np.sin(2 * np.pi * 6.0 * t_long)
        sustained = (0.12 * vibrato * np.sin(2 * np.pi * 440.0 * t_long)).astype(np.float32)
        pulses_1 = detect_vocalise_syllable_pulses(sustained, 0.0, 2.0, 0.005)
        self.assertEqual(len(pulses_1), 1)

        # 2. Two distinct vocal pulses ("Ah — ah": 0.1-0.8s @ 390Hz, dip 0.8-1.0s, 1.0-1.8s @ 291Hz) -> 2 pulses
        two = np.zeros(sr * 2, dtype=np.float32)
        t1 = np.arange(int(0.7 * sr), dtype=np.float32) / sr
        t2 = np.arange(int(0.8 * sr), dtype=np.float32) / sr
        two[int(0.1 * sr) : int(0.8 * sr)] = 0.10 * np.sin(2 * np.pi * 390.0 * t1)
        two[int(1.0 * sr) : int(1.8 * sr)] = 0.10 * np.sin(2 * np.pi * 291.0 * t2)
        pulses_2 = detect_vocalise_syllable_pulses(two, 0.0, 2.0, 0.005)
        self.assertEqual(len(pulses_2), 2)
        self.assertLess(pulses_2[0][1], pulses_2[1][0] + 0.05)

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_collapsed_vocalises_and_adlibs_expand_purely_from_asr_without_fake_defaults(self, windows):
        sr = 16000
        audio = np.zeros(sr * 4, dtype=np.float32)
        t1 = np.arange(int(0.7 * sr), dtype=np.float32) / sr
        t2 = np.arange(int(0.8 * sr), dtype=np.float32) / sr
        audio[int(1.0 * sr) : int(1.7 * sr)] = 0.12 * np.sin(2 * np.pi * 390.0 * t1)
        audio[int(1.9 * sr) : int(2.7 * sr)] = 0.12 * np.sin(2 * np.pi * 291.0 * t2)
        rms, floor = compute_vocal_envelope(audio)
        windows.return_value = [self.window(1.0, 2.75)]

        # 1. Works for ANY vocalise/ad-lib (e.g. "La — da", "Yeah — yeah", "Ah — oh"), not just "Ah"
        for whole, p1, p2, expected in (
            ("Ah.", "Ah.", "Oh.", "Ah, oh"),
            ("La.", "La.", "Da.", "La, da"),
            ("Yeah.", "Yeah.", "Yeah.", "Yeah, yeah"),
        ):
            with self.subTest(expected=expected):
                asr = MagicMock()
                asr.transcribe_chunk.side_effect = [
                    dict(text=whole, words=[dict(word=whole, start=1.9, end=2.2)], language="en"),
                    dict(text=p1, words=[], language="en"),
                    dict(text=p2, words=[], language="en"),
                ]
                result, _ = recover_vocals_with_asr(
                    asr=asr, segments=[], full_vocal_16k=audio,
                    rms_env=rms, vocal_floor=floor, duration=4.0, primary_lang="en", vocal_isolated=True,
                )
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["text"], expected)
                self.assertEqual(len(result[0]["words"]), 2)
                self.assertLess(result[0]["words"][0]["start"], 1.3)
                self.assertGreater(result[0]["words"][1]["start"], 1.75)

        # 2. NEVER fabricates/copies default text onto a pulse when ASR returns empty ""
        asr_partial = MagicMock()
        asr_partial.transcribe_chunk.side_effect = [
            dict(text="Ah.", words=[dict(word="Ah.", start=1.9, end=2.2)], language="en"),
            dict(text="", words=[], language="en"),
            dict(text="Ah.", words=[], language="en"),
        ]
        res_partial, _ = recover_vocals_with_asr(
            asr=asr_partial, segments=[], full_vocal_16k=audio,
            rms_env=rms, vocal_floor=floor, duration=4.0, primary_lang="en", vocal_isolated=True,
        )
        self.assertEqual([s["text"] for s in res_partial], ["Ah."])

    @patch("cover_studio.vocal_recovery.find_adlib_review_windows")
    def test_cancellation_before_and_after_remote_engine_call(self, windows):
        windows.return_value = [self.window()]
        asr = MagicMock()
        with self.assertRaises(Cancelled):
            self.run_recovery(asr, should_stop=lambda: True)
        asr.transcribe_chunk.assert_not_called()
        stopped = False

        def transcribe(*args, **kw):
            nonlocal stopped
            stopped = True
            return dict(text="Hey", words=[])

        asr.transcribe_chunk.side_effect = transcribe
        with self.assertRaises(Cancelled):
            self.run_recovery(asr, should_stop=lambda: stopped)


if __name__ == "__main__":
    unittest.main()
