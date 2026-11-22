"""Unit and API tests for AI Lyrics Transcription & Alignment in Harmony Auralis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from cover_studio.app import create_app
from cover_studio.lyrics_service import (
    LyricsCache,
    _patch_openai_whisper_for_turing_fp32,
    _split_long_regions_at_pauses,
    _split_segment_into_lyric_lines,
    align_user_lyrics_to_anchors,
    annotate_conservative_structure,
    compute_vocal_envelope,
    parse_user_lyrics,
    sanitize_edited_segments,
    segment_vocal_activity,
)


class FakeEngine:
    tsumugi = object()
    device = "cpu"

    def transcribe(self, *args):
        pass

    def unload(self):
        pass

    def describe(self):
        return {"models": [], "score_models": [], "loaded": None, "device": "cpu", "tsumugi": True}


class LyricsServiceUnitTests(unittest.TestCase):
    def test_turing_fp32_patch_eliminates_nan_logits(self):
        import torch
        import whisper.model

        _patch_openai_whisper_for_turing_fp32()
        lin = whisper.model.Linear(16, 16).half()
        x = torch.tensor([[float("nan"), float("inf"), -float("inf")] + [1.0] * 13], dtype=torch.float16)
        out = lin(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_unicode_preservation_and_alignment_for_ja_vi_en_ko_zh(self):
        raw_lyrics = (
            "[Verse 1]\n"
            "夢ならばどれほどよかったでしょう\n"
            "Em ơi Hà Nội phố mùa thu\n"
            "[Chorus]\n"
            "I couldn't wait for you to come\n"
            "사랑해 너를 기억해\n"
            "無中に有を生ず\n"
        )
        parsed = parse_user_lyrics(raw_lyrics)
        self.assertEqual(len(parsed), 5)
        self.assertEqual(parsed[0]["section"], "Verse 1")
        self.assertEqual(parsed[0]["text"], "夢ならばどれほどよかったでしょう")
        self.assertEqual(parsed[1]["text"], "Em ơi Hà Nội phố mùa thu")
        self.assertEqual(parsed[2]["section"], "Chorus")
        self.assertEqual(parsed[3]["text"], "사랑해 너를 기억해")
        self.assertEqual(parsed[4]["text"], "無中に有を生ず")

        anchors = [
            {"word": "夢ならば", "start": 2.0, "end": 4.0, "probability": 0.92},
            {"word": "どれほど", "start": 4.1, "end": 6.0, "probability": 0.89},
            {"word": "Em", "start": 8.0, "end": 8.6, "probability": 0.95},
            {"word": "ơi", "start": 8.6, "end": 9.2, "probability": 0.91},
            {"word": "Hà", "start": 9.3, "end": 9.9, "probability": 0.90},
            {"word": "Nội", "start": 9.9, "end": 10.6, "probability": 0.93},
            {"word": "I", "start": 12.0, "end": 12.4, "probability": 0.96},
            {"word": "couldn't", "start": 12.4, "end": 13.1, "probability": 0.94},
            {"word": "wait", "start": 13.1, "end": 13.8, "probability": 0.95},
        ]
        rms_env = np.full(600, 0.05, dtype=np.float32)
        segments, method = align_user_lyrics_to_anchors(
            user_lines=parsed,
            anchors=anchors,
            rms_env=rms_env,
            vocal_floor=0.002,
            duration=30.0,
            primary_lang="ja",
        )
        self.assertEqual(len(segments), 5)
        # Mode 2 guarantees user-provided text is the source of truth
        self.assertEqual(segments[0]["text"], "夢ならばどれほどよかったでしょう")
        self.assertEqual(segments[1]["text"], "Em ơi Hà Nội phố mùa thu")
        self.assertEqual(segments[1]["lang"], "vi")
        self.assertEqual(segments[2]["lang"], "en")
        self.assertEqual(segments[3]["lang"], "ko")
        self.assertIn(method, ("word", "line"))

    def test_reference_alignment_keeps_repeated_lines_in_their_audio_order(self):
        """Repeated chorus lines must map to each occurrence, never the first occurrence twice."""
        user_lines = parse_user_lyrics("We go home\nWe go home")
        anchors = [
            {"word": "We", "start": 1.0, "end": 1.2, "probability": 0.98},
            {"word": "go", "start": 1.2, "end": 1.5, "probability": 0.98},
            {"word": "home", "start": 1.5, "end": 2.0, "probability": 0.98},
            # A word between two identical lyric occurrences is common in ASR output.
            {"word": "yeah", "start": 4.0, "end": 4.3, "probability": 0.92},
            {"word": "We", "start": 8.0, "end": 8.2, "probability": 0.98},
            {"word": "go", "start": 8.2, "end": 8.5, "probability": 0.98},
            {"word": "home", "start": 8.5, "end": 9.0, "probability": 0.98},
        ]
        segments, _ = align_user_lyrics_to_anchors(
            user_lines=user_lines,
            anchors=anchors,
            rms_env=np.full(240, 0.05, dtype=np.float32),
            vocal_floor=0.002,
            duration=12.0,
            primary_lang="en",
        )
        self.assertEqual([s["text"] for s in segments], ["We go home", "We go home"])
        self.assertLess(segments[0]["start"], 2.5)
        self.assertGreater(segments[1]["start"], 7.0)
        self.assertLessEqual(segments[0]["end"], segments[1]["start"])

    def test_reference_alignment_retains_user_words_when_asr_misses_one(self):
        """A missed ASR token must not delete a verified user lyric from word-sync output."""
        user_lines = parse_user_lyrics("I will always remember")
        anchors = [
            {"word": "I", "start": 2.0, "end": 2.3, "probability": 0.98},
            {"word": "always", "start": 3.0, "end": 3.6, "probability": 0.95},
            {"word": "remember", "start": 3.7, "end": 4.5, "probability": 0.96},
        ]
        segments, _ = align_user_lyrics_to_anchors(
            user_lines=user_lines,
            anchors=anchors,
            rms_env=np.full(200, 0.05, dtype=np.float32),
            vocal_floor=0.002,
            duration=10.0,
            primary_lang="en",
        )
        self.assertEqual(segments[0]["text"], "I will always remember")
        words = segments[0].get("words") or []
        self.assertEqual([w["word"].strip() for w in words], ["I", "will", "always", "remember"])
        self.assertTrue(all(words[i]["start"] <= words[i + 1]["start"] for i in range(len(words) - 1)))
        self.assertGreaterEqual(words[1]["start"], words[0]["end"])
        self.assertLessEqual(words[1]["end"], words[2]["start"])

    def test_reference_alignment_word_timestamps_are_monotonic_and_inside_line(self):
        user_lines = parse_user_lyrics("Take me home\nTonight we sing")
        anchors = [
            {"word": "Take", "start": 5.0, "end": 5.4, "probability": 0.99},
            {"word": "me", "start": 5.5, "end": 5.7, "probability": 0.99},
            {"word": "home", "start": 5.8, "end": 6.3, "probability": 0.99},
            {"word": "Tonight", "start": 8.0, "end": 8.4, "probability": 0.99},
            {"word": "we", "start": 8.5, "end": 8.7, "probability": 0.99},
            {"word": "sing", "start": 8.8, "end": 9.2, "probability": 0.99},
        ]
        segments, _ = align_user_lyrics_to_anchors(
            user_lines=user_lines,
            anchors=anchors,
            rms_env=np.full(240, 0.05, dtype=np.float32),
            vocal_floor=0.002,
            duration=12.0,
            primary_lang="en",
        )
        self.assertEqual(len(segments), 2)
        self.assertLessEqual(segments[0]["end"], segments[1]["start"])
        for seg in segments:
            self.assertGreaterEqual(seg["start"], 0.0)
            self.assertLess(seg["start"], seg["end"])
            words = seg.get("words") or []
            self.assertTrue(words)
            self.assertGreaterEqual(words[0]["start"], seg["start"])
            self.assertLessEqual(words[-1]["end"], seg["end"] + 0.02)
            self.assertTrue(all(a["start"] <= a["end"] for a in words))
            self.assertTrue(all(words[i]["start"] <= words[i + 1]["start"] for i in range(len(words) - 1)))

    def test_reference_alignment_matches_cjk_characters_to_multi_character_asr_tokens(self):
        # Japanese references are tokenized per character, while Whisper often
        # returns a whole phrase in one acoustic token. Each reference character
        # should retain a slice of that token's measured interval.
        japanese = "".join(map(chr, [0x5922, 0x306A, 0x3089, 0x3070, 0x3069, 0x308C, 0x307B, 0x3069]))
        mixed = "".join(map(chr, [0x5922, 0x306A, 0x3089, 0x3070])) + " tonight"
        rms = np.full(240, 0.05, dtype=np.float32)

        segments, method = align_user_lyrics_to_anchors(
            user_lines=parse_user_lyrics(japanese),
            anchors=[
                {"word": "".join(map(chr, [0x5922, 0x306A, 0x3089, 0x3070])), "start": 2.0, "end": 4.0, "probability": 0.94},
                {"word": "".join(map(chr, [0x3069, 0x308C, 0x307B, 0x3069])), "start": 4.1, "end": 6.1, "probability": 0.94},
            ],
            rms_env=rms,
            vocal_floor=0.002,
            duration=12.0,
            primary_lang="ja",
        )
        self.assertEqual(method, "word")
        self.assertEqual(segments[0]["text"], japanese)
        words = segments[0]["words"]
        self.assertEqual(len(words), 8)
        self.assertAlmostEqual(words[0]["start"], 2.0, places=2)
        self.assertGreaterEqual(words[4]["start"], words[3]["end"])
        self.assertAlmostEqual(words[-1]["end"], 6.1, places=2)

        mixed_segments, mixed_method = align_user_lyrics_to_anchors(
            user_lines=parse_user_lyrics(mixed),
            anchors=[
                {"word": "".join(map(chr, [0x5922, 0x306A, 0x3089, 0x3070])), "start": 3.0, "end": 5.0, "probability": 0.94},
                {"word": "tonight", "start": 5.2, "end": 6.0, "probability": 0.94},
            ],
            rms_env=rms,
            vocal_floor=0.002,
            duration=12.0,
            primary_lang="ja",
        )
        self.assertEqual(mixed_method, "word")
        mixed_words = mixed_segments[0]["words"]
        self.assertEqual(len(mixed_words), 5)
        self.assertEqual(mixed_words[-1]["word"], "tonight")
        self.assertAlmostEqual(mixed_words[-1]["start"], 5.2, places=2)

    def test_reference_lyrics_parser_accepts_lrc_and_srt_imports(self):
        lrc = "[ti:Demo]\n[00:01.20]First line\n[00:03.50][00:04.00]Second line"
        self.assertEqual(
            [line["text"] for line in parse_user_lyrics(lrc)],
            ["First line", "Second line", "Second line"],
        )
        srt = "1\n00:00:01,000 --> 00:00:03,000\nFirst line\n\n2\n00:00:04,000 --> 00:00:06,000\nSecond line"
        self.assertEqual(
            [line["text"] for line in parse_user_lyrics(srt)],
            ["First line", "Second line"],
        )

    def test_anchor_cache_is_scoped_to_language_and_model(self):
        with tempfile.TemporaryDirectory() as root:
            cache = LyricsCache(Path(root))
            payload = {"language": "en", "anchors": [{"word": "hello", "start": 1.0, "end": 1.2}]}
            cache.save_anchors("sha", "en", payload, model_id="turbo")
            loaded = cache.load_anchors("sha", "en", model_id="turbo")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["anchors"], payload["anchors"])
            self.assertIsNone(cache.load_anchors("sha", "en", model_id="large-v3"))
            self.assertIsNone(cache.load_anchors("sha", "vi", model_id="turbo"))

    def test_instrumental_silence_gate_rejects_silent_passages(self):
        # 10s of silence followed by 10s of active vocal tone
        sr = 16000
        silence = np.zeros(10 * sr, dtype=np.float32)
        t = np.linspace(0, 10.0, 10 * sr, dtype=np.float32)
        singing = 0.25 * np.sin(2 * np.pi * 440 * t)
        audio = np.concatenate([silence, singing])

        rms_env, floor = compute_vocal_envelope(audio)
        silent_mean, silent_ratio = segment_vocal_activity(rms_env, floor, 1.0, 8.0)
        active_mean, active_ratio = segment_vocal_activity(rms_env, floor, 11.0, 18.0)
        self.assertLess(silent_ratio, 0.05)
        self.assertLess(silent_mean, floor)
        self.assertGreater(active_ratio, 0.90)
        self.assertGreater(active_mean, floor * 5)

    def test_conservative_structure_detects_repeated_chorus(self):
        segs = [
            {"id": 1, "start": 2.0, "end": 5.0, "text": "Walking down the quiet street tonight", "section": None},
            {"id": 2, "start": 5.2, "end": 8.5, "text": "Shadows dancing under silver light", "section": None},
            {"id": 3, "start": 12.0, "end": 15.0, "text": "We will fly beyond the stars above", "section": None},
            {"id": 4, "start": 15.2, "end": 18.5, "text": "Holding on to everything we love", "section": None},
            {"id": 5, "start": 24.0, "end": 27.0, "text": "Morning comes and washes out the rain", "section": None},
            {"id": 6, "start": 27.2, "end": 30.5, "text": "Every memory begins again", "section": None},
            {"id": 7, "start": 35.0, "end": 38.0, "text": "We will fly beyond the stars above", "section": None},
            {"id": 8, "start": 38.2, "end": 41.5, "text": "Holding on to everything we love", "section": None},
        ]
        annotated = annotate_conservative_structure(segs, duration=45.0)
        self.assertEqual(annotated[0]["section"], "Verse 1")
        self.assertEqual(annotated[2]["section"], "Chorus")
        self.assertEqual(annotated[6]["section"], "Chorus")

    def test_sanitize_edited_segments_updates_timestamps_cleanly(self):
        edited = sanitize_edited_segments(
            [
                {"id": 9, "start": 4.5, "end": 8.2, "text": "  Câu hát đã chỉnh sửa chuẩn xác  ", "section": "Chorus"},
                {"id": 10, "start": 10.0, "end": 9.0, "text": "Thời gian tự sửa", "section": ""},
            ],
            fallback_lang="vi",
        )
        self.assertEqual(len(edited), 2)
        self.assertEqual(edited[0]["id"], 1)
        self.assertEqual(edited[0]["text"], "Câu hát đã chỉnh sửa chuẩn xác")
        self.assertEqual(edited[0]["section"], "Chorus")
        self.assertGreater(edited[1]["end"], edited[1]["start"])

    def test_split_long_regions_and_multi_clause_lyric_lines(self):
        # 60s continuous vocal region with breath pause at 18.0s and 36.0s
        rms = np.full(1200, 0.08, dtype=np.float32)
        rms[355:365] = 0.001  # dip around 18.0s
        rms[715:725] = 0.001  # dip around 36.0s
        chunks = _split_long_regions_at_pauses([(0.0, 60.0)], rms, 60.0)
        self.assertGreaterEqual(len(chunks), 3)
        for s_t, e_t in chunks:
            self.assertLessEqual(e_t - s_t, 24.0)

        words = [
            {"word": " I", "start": 22.4, "end": 23.3, "probability": 0.99, "uncertain": False},
            {"word": " woke", "start": 23.3, "end": 23.6, "probability": 0.99, "uncertain": False},
            {"word": " up", "start": 23.6, "end": 23.9, "probability": 0.99, "uncertain": False},
            {"word": " to", "start": 23.9, "end": 24.1, "probability": 0.99, "uncertain": False},
            {"word": " the", "start": 24.1, "end": 24.4, "probability": 0.99, "uncertain": False},
            {"word": " fire,", "start": 24.4, "end": 25.1, "probability": 0.98, "uncertain": False},
            {"word": " a", "start": 25.3, "end": 25.4, "probability": 0.97, "uncertain": False},
            {"word": " quiet", "start": 25.4, "end": 25.9, "probability": 0.98, "uncertain": False},
            {"word": " world", "start": 25.9, "end": 26.3, "probability": 0.99, "uncertain": False},
            {"word": " that", "start": 26.3, "end": 26.5, "probability": 0.98, "uncertain": False},
            {"word": " needed", "start": 26.5, "end": 27.2, "probability": 0.99, "uncertain": False},
            {"word": " just", "start": 27.2, "end": 27.6, "probability": 0.99, "uncertain": False},
            {"word": " a", "start": 27.6, "end": 27.8, "probability": 0.99, "uncertain": False},
            {"word": " little", "start": 27.8, "end": 28.1, "probability": 0.99, "uncertain": False},
            {"word": " extra", "start": 28.1, "end": 28.7, "probability": 0.99, "uncertain": False},
            {"word": " light", "start": 28.7, "end": 29.8, "probability": 0.99, "uncertain": False},
        ]
        sub_lines = _split_segment_into_lyric_lines(
            "I woke up to the fire, a quiet world that needed just a little extra light",
            22.4,
            29.8,
            words,
        )
        self.assertEqual(len(sub_lines), 2)
        self.assertEqual(sub_lines[0][0], "I woke up to the fire")
        self.assertEqual(sub_lines[1][0], "A quiet world that needed just a little extra light")
        self.assertAlmostEqual(sub_lines[1][1], 25.3, places=1)

        # Verify CJK (Japanese) multi-clause segment splitting at full-width spaces / breath pauses
        ja_words = [
            {"word": "夢", "start": 10.0, "end": 10.6, "probability": 0.99, "uncertain": False},
            {"word": "ならば", "start": 10.6, "end": 11.4, "probability": 0.98, "uncertain": False},
            {"word": "どれほど", "start": 11.4, "end": 12.3, "probability": 0.99, "uncertain": False},
            {"word": "よかったでしょう ", "start": 12.3, "end": 13.5, "probability": 0.97, "uncertain": False},
            {"word": "未だに", "start": 13.9, "end": 14.7, "probability": 0.98, "uncertain": False},
            {"word": "あなたのことを", "start": 14.7, "end": 15.8, "probability": 0.99, "uncertain": False},
            {"word": "夢にみる", "start": 15.8, "end": 17.0, "probability": 0.99, "uncertain": False},
        ]
        ja_sub = _split_segment_into_lyric_lines(
            "夢ならばどれほどよかったでしょう 未だにあなたのことを夢にみる",
            10.0,
            17.0,
            ja_words,
        )
        self.assertEqual(len(ja_sub), 2)
        self.assertEqual(ja_sub[0][0], "夢ならばどれほどよかったでしょう")
        self.assertEqual(ja_sub[1][0], "未だにあなたのことを夢にみる")


class LyricsApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app(Path(self.temp.name), FakeEngine())
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_midi_only_project_rejects_lyrics_extraction_gracefully(self):
        res = self.client.post(
            "/api/projects",
            files={"midi": ("piano_only.mid", b"MThd", "audio/midi")},
        )
        self.assertEqual(res.status_code, 200)
        pid = res.json()["id"]

        l_get = self.client.get(f"/api/projects/{pid}/lyrics")
        self.assertEqual(l_get.status_code, 200)
        self.assertFalse(l_get.json()["has_audio"])

        l_ext = self.client.post(
            f"/api/projects/{pid}/lyrics/extract",
            json={"language": "auto", "model": "turbo", "isolate_vocals": True},
        )
        self.assertEqual(l_ext.status_code, 400)
        self.assertIn("MIDI", l_ext.json()["detail"])

    def test_project_lyrics_save_and_persistence(self):
        res = self.client.post(
            "/api/projects",
            data={"title": "Test Song"},
            files={"audio": ("song.mp3", b"fake-mp3-bytes", "audio/mpeg")},
        )
        self.assertEqual(res.status_code, 200)
        pid = res.json()["id"]

        save_res = self.client.put(
            f"/api/projects/{pid}/lyrics",
            json={
                "segments": [
                    {"id": 1, "start": 1.2, "end": 4.5, "text": "夢ならばどれほどよかったでしょう", "lang": "ja", "section": "Verse 1"},
                    {"id": 2, "start": 5.0, "end": 9.1, "text": "Em ơi Hà Nội phố", "lang": "vi", "section": "Chorus"},
                ]
            },
        )
        self.assertEqual(save_res.status_code, 200)
        payload = save_res.json()
        self.assertEqual(payload["state"], "done")
        self.assertEqual(payload["segment_count"], 2)
        self.assertEqual(payload["transcript"]["segments"][0]["text"], "夢ならばどれほどよかったでしょう")
        self.assertEqual(payload["transcript"]["segments"][1]["text"], "Em ơi Hà Nội phố")


if __name__ == "__main__":
    unittest.main()

