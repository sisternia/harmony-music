"""Unit and API tests for AI Lyrics Transcription & Alignment in Harmony Auralis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from cover_studio.app import create_app
from cover_studio.lyrics_schema import (
    SCHEMA_VERSION,
    PipelineTrace,
    classify_vocal_event,
    cleanup_debug_audio,
    compute_pipeline_config_hash,
    estimate_vocal_harmonicity,
    evaluate_transcript_against_reference,
    map_crop_time_to_source,
    migrate_transcript,
    normalize_vocable_elongation,
    stream_label,
)
from cover_studio.lyrics_service import (
    LyricsCache,
    _clean_vocalise_text,
    _split_long_regions_at_pauses,
    _split_segment_into_lyric_lines,
    align_user_lyrics_to_anchors,
    annotate_conservative_structure,
    compute_vocal_envelope,
    parse_user_lyrics,
    sanitize_edited_segments,
    segment_vocal_activity,
)
from cover_studio.music_asr_engine import (
    _dtw_path,
    _flag_untranscribed_vocal_chunk,
    _format_whisper_chunks,
    _group_words_into_lyric_lines,
    iso_to_whisper_lang,
    normalize_model_id,
    whisper_lang_to_iso,
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
    def test_music_asr_model_registry_and_whisper_word_grouping(self):
        # Legacy model IDs upgrade to Qwen; explicitly selected Whisper remains available.
        self.assertEqual(normalize_model_id("turbo"), "qwen3-asr-1.7b")
        self.assertEqual(normalize_model_id("large-v3"), "qwen3-asr-1.7b")
        self.assertEqual(normalize_model_id("hybrid"), "qwen3-asr-1.7b")
        self.assertEqual(normalize_model_id("heart-transcriptor"), "qwen3-asr-1.7b")
        self.assertEqual(normalize_model_id("qwen3-asr-1.7b"), "qwen3-asr-1.7b")
        self.assertEqual(normalize_model_id("gemini-songscription"), "qwen3-asr-1.7b")
        self.assertEqual(normalize_model_id("whisper-large-v3"), "whisper-large-v3")

        # Whisper Large v3 language mapping
        self.assertEqual(iso_to_whisper_lang("en"), "english")
        self.assertEqual(iso_to_whisper_lang("vi"), "vietnamese")
        self.assertIsNone(iso_to_whisper_lang("auto"))
        self.assertEqual(whisper_lang_to_iso("<|ja|>"), "ja")
        self.assertEqual(whisper_lang_to_iso("japanese"), "ja")

        # Whisper word chunk formatting & pause-based line grouping
        raw_chunks = [
            {"text": " So", "timestamp": (0.10, 0.50)},
            {"text": " get", "timestamp": (0.52, 0.90)},
            {"text": " away", "timestamp": (0.92, 1.60)},
            {"text": " Another", "timestamp": (2.35, 2.85)},
            {"text": " way", "timestamp": (2.88, 3.30)},
        ]
        words = _format_whisper_chunks(raw_chunks, offset_sec=10.0, max_duration=60.0, chunk_duration=4.0)
        self.assertEqual(len(words), 5)
        self.assertEqual(words[0]["start"], 10.10)
        lines = _group_words_into_lyric_lines(words, "So get away Another way", 10.0, 14.0)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0][0], "So get away")
        self.assertEqual(lines[1][0], "Another way")

        # Monotonic DTW alignment path recovers diagonal ridge
        sim = np.zeros((4, 12), dtype=np.float64)
        for i in range(4):
            sim[i, i * 3 : (i + 1) * 3] = 5.0
        ti, fi = _dtw_path(-sim)
        self.assertGreater(ti.size, 0)
        self.assertEqual(int(ti[0]), 0)
        self.assertEqual(int(ti[-1]), 3)
        self.assertTrue(np.all(np.diff(fi) >= 0))

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
            self.assertLessEqual(e_t - s_t, 32.5)

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


class SongscriptionVocaliseAndMashupTests(unittest.TestCase):
    """Comprehensive technical test suite covering Section 9 scenarios:
    short-word ad-lib, sustained 'ooh', repeated 'la', quiet/reverbed vocal, vocal-like instrument,
    silence, bilingual mashup, overlapping multi-stream lyrics, lead+backing duplicate, crossfade
    transition, chunk timestamp mapping, legacy migration + user edit preservation, and metrics.
    """

    def test_01_short_word_adlib_and_parenthetical_preserved(self):
        for raw in ("Yeah!", "(yeah)", "Come on!", "Hey, hey"):
            cleaned, is_adlib = _clean_vocalise_text(raw, "en", allow_rhythmic=True)
            self.assertIsNotNone(cleaned, f"Short ad-lib must not be dropped: {raw}")
            ev_type, ev_adlib, _ = classify_vocal_event(cleaned)
            self.assertTrue(ev_adlib or is_adlib)
            self.assertIn(ev_type, ("lyric", "vocable"))

    def test_02_sustained_ooh_normalized_without_char_stretching(self):
        # Duration belongs in start/end timestamps, never 20 repeated letters!
        self.assertEqual(normalize_vocable_elongation("Ooooooooooooooh"), "Ooh")
        self.assertEqual(normalize_vocable_elongation("Aaaaaaaaaah"), "Aah")
        self.assertEqual(normalize_vocable_elongation("Mmmmmmmmm"), "Mmm")
        cleaned, is_adlib = _clean_vocalise_text("Ooooooooooooooh", "en", allow_rhythmic=True)
        self.assertEqual(cleaned, "Ooh")
        self.assertTrue(is_adlib)
        ev_type, _, lang_hint = classify_vocal_event(cleaned, fallback_lang="en")
        self.assertEqual(ev_type, "vocable")
        # Ambiguous single vowel vocalise should NOT force a language
        self.assertIsNone(lang_hint)

    def test_03_repeated_la_sequence_preserved_without_collapse(self):
        cleaned, is_adlib = _clean_vocalise_text("La, la, la, la", "en", comp_ratio=3.1, allow_rhythmic=True)
        self.assertEqual(cleaned, "La, la, la, la")
        self.assertTrue(is_adlib)
        # Rhythmic scat pattern preserves syllable variety
        scat, is_scat = _clean_vocalise_text("Ba ee ya ee ba ba ee ya ee", "en", comp_ratio=3.6, allow_rhythmic=True)
        self.assertEqual(scat, "Ba ee ya ee ba ba ee ya ee")
        self.assertTrue(is_scat)

    def test_04_quiet_reverbed_vocal_becomes_honest_unknown_vocal_not_fake_ah_oh(self):
        # Synthetic harmonic vocal-like singing tone (F0=260Hz + harmonics + gentle decay)
        sr = 16000
        dur = 8.0
        t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
        audio = np.zeros_like(t)
        # Active singing region [2.0, 5.5]
        mask = (t >= 2.0) & (t <= 5.5)
        tm = t[mask]
        audio[mask] = (
            0.06 * np.sin(2 * np.pi * 260 * tm)
            + 0.04 * np.sin(2 * np.pi * 520 * tm)
            + 0.025 * np.sin(2 * np.pi * 780 * tm)
            + 0.015 * np.sin(2 * np.pi * 1040 * tm)
        )
        rms, floor = compute_vocal_envelope(audio)

        trace = PipelineTrace(model_id="qwen3-asr-1.7b", requested_lang="auto", vocal_isolated=True, audio_sha1="abc")
        seg = _flag_untranscribed_vocal_chunk(
            chunk_audio=audio[int(2.0 * sr) : int(5.5 * sr)],
            start=2.0,
            end=5.5,
            rms_env=rms,
            vocal_floor=floor,
            stream_id="stream_a",
            trace=trace,
        )
        self.assertIsNotNone(seg)
        assert seg is not None
        self.assertEqual(seg["event_type"], "unknown_vocal")
        # Must NEVER fabricate "Ah" / "Oh" or fake percentage confidence!
        self.assertEqual(seg["text"], "")
        self.assertIsNone(seg["language"])
        self.assertIsNone(seg["confidence"])
        self.assertEqual(seg["review_status"], "needs_review")
        self.assertIn("suspected_vocal_untranscribed", seg["review_reasons"])
        self.assertEqual(seg["timing_granularity"], "region")
        self.assertEqual(seg["words"], [])
        self.assertTrue(len(trace.recovery_probes) >= 1)

    def test_05_non_vocal_white_noise_and_silence_rejected(self):
        sr = 16000
        # Flat white noise (high spectral flatness, zero pitch voicing) should score low vocal_likely
        rng = np.random.default_rng(42)
        noise = (0.08 * rng.standard_normal(sr * 3)).astype(np.float32)
        harm_noise = estimate_vocal_harmonicity(noise)
        self.assertLess(harm_noise["vocal_likely"], 0.40)

        # Pure silence -> zero vocal activity
        silence = np.zeros(sr * 4, dtype=np.float32)
        rms, floor = compute_vocal_envelope(silence)
        mr, ar = segment_vocal_activity(rms, floor, 0.5, 3.5)
        self.assertEqual(ar, 0.0)
        self.assertLess(mr, floor)

    def test_06_bilingual_mashup_preserves_both_languages_when_mixed(self):
        # Japanese line inside an English-majority bilingual mashup must NOT be dropped!
        ja_txt, _ = _clean_vocalise_text("夢ならばどれほどよかったでしょう", "en", is_mixed=True)
        self.assertEqual(ja_txt, "夢ならばどれほどよかったでしょう")
        ko_txt, _ = _clean_vocalise_text("사랑해 너를 기억해", "en", is_mixed=True)
        self.assertEqual(ko_txt, "사랑해 너를 기억해")
        # Song-level language and repeated characters cannot establish a hallucination.
        spam, _ = _clean_vocalise_text("爱爱爱爱爱爱爱爱爱爱", "en", is_mixed=False)
        self.assertEqual(spam, "爱爱爱爱爱爱爱爱爱爱")

    def test_07_overlapping_dual_streams_preserved_with_overlap_group_id(self):
        segs = [
            {"id": 1, "start": 10.0, "end": 14.0, "text": "Lead melody line", "stream_id": "stream_a", "lang": "en"},
            {"id": 2, "start": 11.0, "end": 14.5, "text": "Counter melody backing", "stream_id": "stream_b", "lang": "ja"},
            {"id": 3, "start": 16.0, "end": 19.0, "text": "Solo line after", "stream_id": "stream_a", "lang": "en"},
        ]
        sanitized = sanitize_edited_segments(segs, fallback_lang="en")
        self.assertEqual(len(sanitized), 3)
        # Overlapping timestamps MUST NOT be clamped apart or flattened!
        self.assertEqual(sanitized[0]["start"], 10.0)
        self.assertEqual(sanitized[1]["start"], 11.0)
        self.assertIsNotNone(sanitized[0]["overlap_group_id"])
        self.assertEqual(sanitized[0]["overlap_group_id"], sanitized[1]["overlap_group_id"])
        self.assertIsNone(sanitized[2]["overlap_group_id"])
        self.assertEqual(stream_label(sanitized[0]["stream_id"]), "Luồng A")
        self.assertEqual(stream_label(sanitized[1]["stream_id"]), "Luồng B")

    def test_08_lead_and_backing_same_text_on_separate_streams_both_kept(self):
        segs = [
            {"id": 1, "start": 20.0, "end": 23.0, "text": "We rise together", "stream_id": "stream_a", "vocal_role": "lead"},
            {"id": 2, "start": 20.1, "end": 23.1, "text": "We rise together", "stream_id": "stream_b", "vocal_role": "backing"},
        ]
        sanitized = sanitize_edited_segments(segs, fallback_lang="en")
        self.assertEqual(len(sanitized), 2)
        self.assertEqual(sanitized[0]["stream_id"], "stream_a")
        self.assertEqual(sanitized[1]["stream_id"], "stream_b")
        self.assertEqual(sanitized[0]["overlap_group_id"], sanitized[1]["overlap_group_id"])

    def test_09_user_lyrics_stream_headers_parsed_for_mashup_alignment(self):
        raw = "[Luồng A]\n[Verse 1]\nFirst song line\n[Luồng B]\nSecond stream line"
        parsed = parse_user_lyrics(raw)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["stream_id"], "stream_a")
        self.assertEqual(parsed[0]["section"], "Verse 1")
        self.assertEqual(parsed[1]["stream_id"], "stream_b")

    def test_10_timestamp_mapping_applies_offset_once_across_chunk_boundaries(self):
        s, e = map_crop_time_to_source(1.25, 3.80, crop_offset=42.50, source_duration=180.0)
        self.assertAlmostEqual(s, 43.75, places=2)
        self.assertAlmostEqual(e, 46.30, places=2)
        # Passing already_mapped=True must NOT double-add offset!
        s2, e2 = map_crop_time_to_source(s, e, crop_offset=42.50, source_duration=180.0, already_mapped=True)
        self.assertAlmostEqual(s2, 43.75, places=2)
        self.assertAlmostEqual(e2, 46.30, places=2)

    def test_11_legacy_v1_migration_preserves_user_edits_and_immutable_raw(self):
        legacy_v1 = {
            "mode": "extract",
            "language": "en",
            "duration": 30.0,
            "segments": [
                {
                    "id": 1,
                    "start": 2.0,
                    "end": 5.0,
                    "text": "User corrected lyric",
                    "lang": "en",
                    "review_status": "edited",
                    "confidence": 0.91,
                    "low_confidence": False,
                    "words": [],
                },
                {
                    "id": 2,
                    "start": 6.0,
                    "end": 8.0,
                    "text": "Hmm",
                    "lang": "en",
                    "confidence": 0.72,
                    "low_confidence": False,
                    "words": [{"word": "Hmm", "start": 6.0, "end": 8.0, "probability": 0.7}],
                },
            ],
        }
        migrated = migrate_transcript(legacy_v1)
        self.assertEqual(migrated["schema_version"], SCHEMA_VERSION)
        self.assertEqual(migrated["segments"][0]["text"], "User corrected lyric")
        self.assertEqual(migrated["segments"][0]["review_status"], "edited")
        self.assertEqual(migrated["segments"][0]["text_raw"], "User corrected lyric")
        # Humming must have region granularity, no forced word alignment, and language=None
        self.assertEqual(migrated["segments"][1]["event_type"], "humming")
        self.assertIsNone(migrated["segments"][1]["language"])
        self.assertEqual(migrated["segments"][1]["words"], [])
        self.assertEqual(migrated["segments"][1]["timing_granularity"], "region")
        # Editing keeps text_raw immutable and updates text_edited
        edited = sanitize_edited_segments(
            [{**migrated["segments"][0], "text": "Second manual edit"}],
            fallback_lang="en",
        )
        self.assertEqual(edited[0]["text_raw"], "User corrected lyric")
        self.assertEqual(edited[0]["text_edited"], "Second manual edit")

    def test_12_config_scoped_cache_and_debug_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = LyricsCache(root)
            h1 = compute_pipeline_config_hash(model_id="large-v3", requested_lang="auto", vocal_isolated=True, asr_params={"beam": 5})
            h2 = compute_pipeline_config_hash(model_id="large-v3", requested_lang="auto", vocal_isolated=True, asr_params={"beam": 8})
            self.assertNotEqual(h1, h2)
            cache.save_anchors("sha1", "auto", {"anchors": [{"word": "hi"}]}, model_id="large-v3", vocal_isolated=True, config_hash=h1)
            self.assertIsNotNone(cache.load_anchors("sha1", "auto", model_id="large-v3", vocal_isolated=True, config_hash=h1))
            self.assertIsNone(cache.load_anchors("sha1", "auto", model_id="large-v3", vocal_isolated=True, config_hash=h2))

            # Bounded debug cleanup prunes excess directories
            dbg = root / "debug_audio"
            for i in range(8):
                (dbg / f"sess_{i}").mkdir(parents=True)
            removed = cleanup_debug_audio(dbg, max_sessions=3, max_age_hours=24.0)
            self.assertEqual(removed, 5)
            self.assertEqual(len(list(dbg.iterdir())), 3)

    def test_13_evaluation_metrics_do_not_inflate_unknown_vocal(self):
        ref = [
            {"start": 1.0, "end": 3.0, "text": "hello world", "event_type": "lyric", "is_adlib": False, "stream_id": "stream_a"},
            {"start": 4.0, "end": 6.0, "text": "Ooh", "event_type": "vocable", "is_adlib": True, "stream_id": "stream_a"},
            {"start": 7.0, "end": 9.0, "text": "La la", "event_type": "vocable", "is_adlib": True, "stream_id": "stream_b"},
        ]
        pred = [
            {"start": 1.1, "end": 3.0, "text": "hello world", "event_type": "lyric", "is_adlib": False, "stream_id": "stream_a"},
            {"start": 4.0, "end": 6.0, "text": "Ooh", "event_type": "vocable", "is_adlib": True, "stream_id": "stream_a"},
            # unknown_vocal MUST NOT count as correctly recognized vocalise!
            {"start": 7.0, "end": 9.0, "text": "", "event_type": "unknown_vocal", "is_adlib": None, "stream_id": "stream_a"},
        ]
        metrics = evaluate_transcript_against_reference(pred, ref, silent_intervals=[(12.0, 16.0)])
        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["vocalise_precision"], 1.0)
        self.assertEqual(metrics["vocalise_recall"], 0.5)
        self.assertEqual(metrics["adlib_miss_rate"], 0.5)
        self.assertEqual(metrics["unknown_vocal_count"], 1)
        self.assertEqual(metrics["hallucination_count"], 0)

    def test_16_runaway_word_loop_rejected_and_common_words_not_vocables(self):
        # 20x "No, no, no..." Whisper decoder loop MUST be dropped, never rescued as a vocable!
        spam = ", ".join(["No"] * 20)
        cleaned, _ = _clean_vocalise_text(spam, "en", comp_ratio=6.5, allow_rhythmic=True)
        self.assertIsNone(cleaned)
        # Common word "So..." is a lyric, not a vocable
        so_txt, so_adlib = _clean_vocalise_text("So...", "en")
        self.assertEqual(so_txt, "So...")
        self.assertFalse(so_adlib)

    def test_17_vocalise_never_split_on_commas_and_snapped_to_real_acoustic_bursts(self):
        from cover_studio.lyrics_service import (
            _align_vocalise_to_envelope,
            _is_runaway_token_loop,
            _split_segment_into_lyric_lines,
            _validate_words,
        )
        from cover_studio.music_asr_engine import _enforce_non_overlapping_same_stream

        # 1. Authentic repeated vocalises keep their exact count (even 40x "Oh");
        #    only physically impossible decoder runaway loops (e.g. 200x "ah" in 24s) are rejected.
        self.assertFalse(_is_runaway_token_loop(", ".join(["Oh"] * 40), comp_ratio=26.2))
        self.assertFalse(_is_runaway_token_loop("Oh, oh, oh, oh", comp_ratio=2.1))
        self.assertTrue(_is_runaway_token_loop("On wide horizons, " + " ".join(["ah"] * 200), duration_sec=24.4))

        # 2. Pure vocalise phrase is NOT chopped on commas into tiny lines
        words = [
            {"word": " Oh,", "start": 10.0 + i * 0.6, "end": 10.5 + i * 0.6, "probability": 0.9}
            for i in range(8)
        ]
        parts = _split_segment_into_lyric_lines("Oh, oh, oh, oh, oh, oh, oh, oh", 10.0, 15.0, words)
        self.assertEqual(len(parts), 1)

        # 3. Collapsed identical word timestamps (e.g. 3 words all stamped 174.28-174.33) are rejected
        collapsed = [
            {"word": " oh,", "start": 173.84, "end": 174.02, "probability": 0.99},
            {"word": " oh,", "start": 174.28, "end": 174.33, "probability": 0.99},
            {"word": " oh,", "start": 174.28, "end": 174.33, "probability": 0.99},
            {"word": " oh,", "start": 174.28, "end": 174.33, "probability": 0.99},
        ]
        self.assertEqual(_validate_words(collapsed, 173.84, 174.33), [])

        # 4. Without word timing, bursts cannot justify copying the whole phrase twice.
        rms_env = np.zeros(400, dtype=np.float32)  # 20s at 20 fps
        rms_env[40:100] = 0.08   # Burst 1: 2.0s - 5.0s
        rms_env[160:220] = 0.08  # Burst 2: 8.0s - 11.0s (3s silence in between!)
        seg = {
            "start": 1.0,
            "end": 12.0,
            "text": "Oh, oh, oh",
            "event_type": "vocable",
            "is_adlib": True,
            "stream_id": "stream_a",
            "words": [],
        }
        aligned = _align_vocalise_to_envelope(seg, rms_env, 0.004, 20.0)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0]["text"], "Oh, oh, oh")
        self.assertEqual(aligned[0]["review_status"], "needs_review")

        # 5. Every emitted ad-lib/vocalise occurrence survives, including identical coarse
        # timestamps. Concurrent occurrences receive separate playback lanes.
        dups = [
            {"start": 168.0, "end": 170.0, "text": "Oh, oh, oh", "event_type": "vocable", "stream_id": "stream_a"},
            {"start": 168.0, "end": 170.0, "text": "Oh, oh, oh", "event_type": "vocable", "stream_id": "stream_a"},
            {"start": 168.04, "end": 170.02, "text": "Oh, oh, oh", "event_type": "vocable", "stream_id": "stream_a"},
            {"start": 170.1, "end": 171.5, "text": "Oh, oh, oh", "event_type": "vocable", "stream_id": "stream_a"},
            {"start": 174.66, "end": 174.81, "text": "Oh, oh, oh", "event_type": "vocable", "stream_id": "stream_a"},
        ]
        cons = _enforce_non_overlapping_same_stream(dups)
        self.assertEqual(len(cons), 5)
        self.assertEqual((cons[0]["start"], cons[0]["end"]), (168.0, 170.0))
        self.assertEqual((cons[1]["start"], cons[1]["end"]), (168.0, 170.0))
        self.assertEqual((cons[2]["start"], cons[2]["end"]), (168.04, 170.02))
        self.assertEqual((cons[3]["start"], cons[3]["end"]), (170.1, 171.5))
        self.assertEqual((cons[4]["start"], cons[4]["end"]), (174.66, 174.81))
        self.assertEqual(len({seg["stream_id"] for seg in cons[:3]}), 3)

    def test_18_sentence_line_breaks_from_whisper_punctuation_and_separate_adlibs(self):
        from cover_studio.music_asr_engine import _group_words_into_lyric_lines

        # 1. Whisper native word chunks preserve punctuation -> 1 sentence per line
        raw_chorus = "So, what do you wanna do? What's your point of view? There's a party soon, do you wanna go?"
        tokens = [
            "So,", "what", "do", "you", "wanna", "do?",
            "What's", "your", "point", "of", "view?",
            "There's", "a", "party", "soon,", "do", "you", "wanna", "go?",
        ]
        t = 40.0
        whisper_words = []
        for idx, tok in enumerate(tokens):
            whisper_words.append(
                {
                    "word": tok if idx == 0 else f" {tok}",
                    "start": round(t, 2),
                    "end": round(t + 0.22, 2),
                    "probability": None,
                    "uncertain": False,
                }
            )
            t += 0.28  # fast chorus: tiny 0.06s gaps between words
        lines = _group_words_into_lyric_lines(whisper_words, raw_chorus, 40.0, t)
        self.assertEqual(
            [ln[0] for ln in lines],
            [
                "So, what do you wanna do?",
                "What's your point of view?",
                "There's a party soon, do you wanna go?",
            ],
        )
        # Every line's timestamp matches its first..last word tightly (real-time karaoke sync)
        for _txt, ls, le, lw in lines:
            self.assertEqual(ls, lw[0]["start"])
            self.assertEqual(le, lw[-1]["end"])

        # 2. Even WITHOUT punctuation, rapid-fire chorus never builds a 20+ word wall or strands dangling "of"/"to"
        no_punct_toks = [
            "I", "really", "wanna", "stay", "at", "your", "house",
            "and", "I", "hope", "this", "all", "works", "out",
            "but", "you", "know", "how", "much", "you", "broke", "me", "apart",
        ]
        t2 = 60.0
        no_punct_w = []
        for idx, tok in enumerate(no_punct_toks):
            no_punct_w.append({"word": tok if idx == 0 else f" {tok}", "start": round(t2, 2), "end": round(t2 + 0.25, 2)})
            t2 += 0.30
        split_np = _group_words_into_lyric_lines(no_punct_w, " ".join(no_punct_toks), 60.0, t2)
        self.assertGreaterEqual(len(split_np), 2)
        for ln_txt, _ls, _le, lw in split_np:
            self.assertLessEqual(len(lw), 13)
            self.assertNotIn(ln_txt.split()[-1].lower(), {"of", "to", "a", "the", "your"})

        # 3. Separate ad-lib/vocalise phrases separated by >= 0.42s pause stay on separate lines
        voc_words = [
            {"word": "Ooh", "start": 166.0, "end": 167.2},
            {"word": " Oh", "start": 167.8, "end": 168.3},
            {"word": " oh", "start": 168.35, "end": 168.8},
            {"word": " oh", "start": 168.85, "end": 169.4},
        ]
        voc_lines = _group_words_into_lyric_lines(voc_words, "Ooh Oh oh oh", 166.0, 169.4, is_vocalise_chunk=True)
        self.assertEqual(len(voc_lines), 2)
        self.assertEqual(voc_lines[0][0], "Ooh")
        self.assertEqual(voc_lines[1][0], "Oh oh oh")

    def test_19_repair_collapsed_forced_aligner_word_timestamps(self):
        from cover_studio.music_asr_engine import _repair_aligned_words

        # Active singing vocal envelope (20 fps across 0..90s)
        rms_env = np.full(1800, 0.11, dtype=np.float32)

        # Case 1 (Real #26 bug): "go" stalled 3.84s, then 5 words stacked at 78.72-78.77
        stalled = [
            {"word": "Don't", "start": 74.72, "end": 74.88},
            {"word": " wanna", "start": 74.88, "end": 74.93},
            {"word": " go", "start": 74.88, "end": 78.72},
            {"word": " 'cause", "start": 78.72, "end": 78.77},
            {"word": " I", "start": 78.72, "end": 78.77},
            {"word": " really", "start": 78.72, "end": 78.77},
            {"word": " wanna", "start": 78.72, "end": 78.77},
            {"word": " stay", "start": 78.80, "end": 78.85},
        ]
        rep1 = _repair_aligned_words(stalled, rms_env, 0.006, 79.20)
        # "go" must no longer hog 3.84s; every word must be strictly monotonic with realistic duration
        self.assertLess(rep1[2]["end"] - rep1[2]["start"], 1.2)
        for k in range(len(rep1)):
            self.assertGreaterEqual(rep1[k]["end"] - rep1[k]["start"], 0.10)
            if k > 0:
                self.assertGreaterEqual(rep1[k]["start"], rep1[k - 1]["end"])

        # Case 2 (Real #06 bug): "Another" 32.48-32.96 -> 1.36s active hole -> "way to feel." stacked at 34.32
        holed = [
            {"word": "Another", "start": 32.48, "end": 32.96},
            {"word": " way", "start": 34.32, "end": 34.37},
            {"word": " to", "start": 34.32, "end": 34.37},
            {"word": " feel.", "start": 34.32, "end": 34.37},
        ]
        rep2 = _repair_aligned_words(holed, rms_env, 0.006, 34.80)
        # "Another" stays intact; "way" pulls back into the active singing gap right after 32.96
        self.assertEqual((rep2[0]["start"], rep2[0]["end"]), (32.48, 32.96))
        self.assertLess(rep2[1]["start"], 33.20)
        self.assertGreater(rep2[3]["end"], 34.30)
        for k in range(1, len(rep2)):
            self.assertGreaterEqual(rep2[k]["start"], rep2[k - 1]["end"])
            self.assertGreaterEqual(rep2[k]["end"] - rep2[k]["start"], 0.15)

    def test_20_same_stream_never_overlaps_while_cross_stream_preserved(self):
        from cover_studio.music_asr_engine import _enforce_non_overlapping_same_stream

        segs = [
            # Two overlapping lines on stream_a (real bug: line 1 at 22-25s, line 2 starts at 24s)
            {
                "start": 22.0, "end": 25.0, "text": "First line still running", "stream_id": "stream_a",
                "words": [
                    {"word": "First", "start": 22.0, "end": 23.0},
                    {"word": " line", "start": 23.0, "end": 24.2},
                    {"word": " running", "start": 24.2, "end": 25.0},
                ],
            },
            # Engulfed tiny artifact completely inside line 1 on stream_a -> must be dropped
            {"start": 22.5, "end": 23.1, "text": "artifact", "stream_id": "stream_a", "words": []},
            # Next sentence on stream_a starting at 24.0
            {
                "start": 24.0, "end": 26.5, "text": "Second line starts", "stream_id": "stream_a",
                "words": [
                    {"word": "Second", "start": 24.0, "end": 24.8},
                    {"word": " line", "start": 24.8, "end": 25.6},
                    {"word": " starts", "start": 25.6, "end": 26.5},
                ],
            },
            # Genuine simultaneous backing line on stream_b -> MUST keep full overlap with stream_a!
            {"start": 24.2, "end": 26.0, "text": "Ooh yeah", "stream_id": "stream_b", "words": []},
        ]
        fixed = _enforce_non_overlapping_same_stream(segs)
        a_lines = [s for s in fixed if s["stream_id"] == "stream_a"]
        b_lines = [s for s in fixed if s["stream_id"] == "stream_b"]
        self.assertEqual(len(a_lines), 2)
        self.assertEqual(len(b_lines), 1)
        # Line 1 on stream_a & ALL its words MUST finish strictly <= Line 2 start (24.0)
        self.assertLessEqual(a_lines[0]["end"], a_lines[1]["start"])
        self.assertLessEqual(a_lines[0]["words"][-1]["end"], a_lines[1]["start"])
        # stream_b overlap is untouched
        self.assertEqual((b_lines[0]["start"], b_lines[0]["end"]), (24.2, 26.0))

    def test_15_isolated_stream_audio_cache_and_store_sync(self):
        from cover_studio.songscription import SongscriptionStore

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = SongscriptionStore(root)
            src = root / "sample.wav"
            src.write_bytes(b"RIFF" + b"\x00" * 2048)
            item = store.create("Dual Stream Test", src, source_kind="upload")
            sid = item["id"]
            sha1 = item["audio_sha1"]

            # Initially no isolated stream audio files exist
            self.assertIsNone(store.stream_audio_path(sid, "stream_a"))
            self.assertEqual(store.cache.available_streams(sha1), [])

            # Cache separated stream_a and stream_b WAVs
            for st_id in ("stream_a", "stream_b"):
                p = store.cache.stream_vocal_path(sha1, st_id)
                p.write_bytes(b"RIFF" + b"\x01" * 2048)

            self.assertEqual(store.cache.available_streams(sha1), ["stream_a", "stream_b"])
            self.assertEqual(store.stream_audio_path(sid, "stream_a"), store.cache.stream_vocal_path(sha1, "stream_a"))
            self.assertEqual(store.stream_audio_path(sid, "stream_b"), store.cache.stream_vocal_path(sha1, "stream_b"))

            store.update(
                sid,
                transcript={
                    "schema_version": 2,
                    "language": "en",
                    "language_name": "English",
                    "segments": [{"id": 1, "start": 1.0, "end": 3.0, "text": "Lead line", "stream_id": "stream_a"}],
                },
            )
            loaded = store.get(sid)
            stream_map = {s["id"]: s["has_isolated_audio"] for s in loaded["transcript"]["streams"]}
            self.assertTrue(stream_map.get("stream_a"))
            self.assertTrue(stream_map.get("stream_b"))

    def test_22_audio_verified_reference_lyrics_helpers(self):
        from cover_studio.lyrics_reference import (
            build_search_queries,
            estimate_lrc_time_offset,
            parse_synced_lrc,
            verify_reference_against_asr,
        )

        # 1. Noisy YouTube title yields clean title + artist query first
        qs = build_search_queries('Cyberpunk: Edgerunners | “I Really Want to Stay At Your House” by Rosa Walton | Music Video')
        self.assertTrue(qs)
        self.assertIn("I Really Want to Stay At Your House", qs[0])
        self.assertIn("Rosa Walton", qs[0])

        # 2. LRC parsing preserves timestamps and strips inline metadata
        lrc = "[ti:Test]\n[00:01.22] First line here\n[00:08.57] Second line here\n"
        parsed = parse_synced_lrc(lrc)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["start"], 1.22)
        self.assertEqual(parsed[1]["text"], "Second line here")

        # 3. Audio verification gate accepts matching lyrics and rejects unrelated song
        ref_txt = "I couldn't wait for you to come and clear the cupboards but now you're going to leave with nothing but a sign"
        asr_match = "I couldn't wait for you to come and clear the cupboard but now you're gone leaving nothing but a sign"
        asr_wrong = "Yesterday all my troubles seemed so far away now it looks as though they're here to stay"
        ok_good, score_good = verify_reference_against_asr(ref_txt, asr_match)
        ok_bad, score_bad = verify_reference_against_asr(ref_txt, asr_wrong)
        self.assertTrue(ok_good)
        self.assertGreater(score_good, 0.65)
        self.assertFalse(ok_bad)
        self.assertLess(score_bad, 0.25)

        # 4. Constant time offset estimation detects shifted MV intro
        lrc_lines = [
            {"start": 1.0, "text": "I couldn't wait for you to come"},
            {"start": 8.0, "text": "But now you're gone and leaving nothing"},
            {"start": 16.0, "text": "Another evening sitting reading between lines"},
        ]
        asr_segs = [
            {"start": 3.5, "text": "I couldn't wait for you to come"},
            {"start": 10.6, "text": "But now you're gone leaving nothing"},
            {"start": 18.5, "text": "Another evening sitting reading between lines"},
        ]
        offset = estimate_lrc_time_offset(lrc_lines, asr_segs)
        self.assertAlmostEqual(offset, 2.5, delta=0.2)

    def test_24_adlib_acoustic_gap_detection(self):
        from cover_studio.vocal_recovery import find_uncovered_vocal_gaps

        sr = 16000
        dur = 24.0
        audio = np.zeros(int(sr * dur), dtype=np.float32)

        def _add_harmonic_voice(t0: float, t1: float, f0: float = 330.0, amp: float = 0.22) -> None:
            i0, i1 = int(t0 * sr), int(t1 * sr)
            t = np.arange(i1 - i0, dtype=np.float32) / sr
            sig = amp * (0.65 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t) + 0.10 * np.sin(2 * np.pi * 3 * f0 * t))
            audio[i0:i1] += sig

        # Covered main lyric at [1.0 - 4.0] and [19.0 - 22.0]
        _add_harmonic_voice(1.0, 4.0, 290.0)
        _add_harmonic_voice(19.0, 22.0, 310.0)
        # Two distinct uncovered ad-lib phrases separated by a 0.9s breath pause: [6.0 - 8.2] and [9.1 - 11.4]
        _add_harmonic_voice(6.0, 8.2, 440.0)
        _add_harmonic_voice(9.1, 11.4, 520.0)
        # Non-harmonic white noise burst at [13.5 - 15.5] (should be rejected by harmonicity gate)
        rng = np.random.default_rng(42)
        audio[int(13.5 * sr) : int(15.5 * sr)] = 0.10 * rng.standard_normal(int(2.0 * sr)).astype(np.float32)

        rms_env, floor = compute_vocal_envelope(audio)
        existing_segs = [
            {"id": 1, "start": 1.0, "end": 4.0, "text": "Main line one"},
            {"id": 2, "start": 19.0, "end": 22.0, "text": "Main line two"},
        ]
        gaps = find_uncovered_vocal_gaps(
            segments=existing_segs,
            full_vocal_16k=audio,
            rms_env=rms_env,
            vocal_floor=floor,
            duration=dur,
        )
        # Must find BOTH distinct ad-lib spans separately and reject the white-noise span
        self.assertEqual(len(gaps), 2)
        self.assertAlmostEqual(gaps[0]["start"], 6.0, delta=0.25)
        self.assertAlmostEqual(gaps[0]["end"], 8.2, delta=0.25)
        self.assertAlmostEqual(gaps[1]["start"], 9.1, delta=0.25)
        self.assertAlmostEqual(gaps[1]["end"], 11.4, delta=0.25)
        self.assertGreaterEqual(gaps[0]["bound_start"], 4.0)
        self.assertLessEqual(gaps[1]["bound_end"], 19.0)

        # Stretched vocalise variants classify cleanly as vocable / is_adlib=True
        for sample in ("Ohhh", "Aaaah", "Oh-oh-oh", "La-la-la"):
            ev, adlib, _ = classify_vocal_event(sample, fallback_lang="en")
            self.assertEqual(ev, "vocable")
            self.assertTrue(adlib)

    def test_25_instrumental_stem_mixing_derivation_and_api(self):
        from unittest.mock import patch
        import soundfile as sf
        from cover_studio.lyrics_service import convert_audio_ffmpeg, derive_instrumental_from_vocal_cache, mix_non_vocal_stems
        from cover_studio.songscription import SongscriptionStore

        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        vocal_sig = 0.35 * np.sin(2 * np.pi * 440.0 * t)
        inst_l = 0.25 * np.sin(2 * np.pi * 220.0 * t)
        inst_r = 0.25 * np.sin(2 * np.pi * 330.0 * t)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # 1. mix_non_vocal_stems sums non-vocal stems and excludes vocals
            voc_p = root / "vocals.wav"
            bass_p = root / "bass.wav"
            piano_p = root / "piano.wav"
            sf.write(str(voc_p), np.stack([vocal_sig, vocal_sig], axis=1), sr, subtype="PCM_16")
            sf.write(str(bass_p), np.stack([inst_l, np.zeros_like(inst_l)], axis=1), sr, subtype="PCM_16")
            sf.write(str(piano_p), np.stack([np.zeros_like(inst_r), inst_r], axis=1), sr, subtype="PCM_16")

            mixed_p = mix_non_vocal_stems({"vocals": voc_p, "bass": bass_p, "piano": piano_p}, root / "mixed.wav")
            self.assertIsNotNone(mixed_p)
            mixed_arr, _ = sf.read(str(mixed_p), dtype="float32")
            self.assertAlmostEqual(float(np.corrcoef(mixed_arr[:, 0], inst_l)[0, 1]), 1.0, places=3)
            self.assertAlmostEqual(float(np.corrcoef(mixed_arr[:, 1], inst_r)[0, 1]), 1.0, places=3)

            # 2. derive_instrumental_from_vocal_cache removes vocal from stereo mix & preserves music
            mix_stereo = np.stack([inst_l + vocal_sig, inst_r + vocal_sig], axis=1)
            src_wav = root / "source.wav"
            sf.write(str(src_wav), mix_stereo, sr, subtype="PCM_16")
            voc_mono_p = root / "vocal_mono.wav"
            sf.write(str(voc_mono_p), vocal_sig, sr, subtype="PCM_16")

            cache = LyricsCache(root)
            derived = derive_instrumental_from_vocal_cache(src_wav, voc_mono_p, cache, "testsha1")
            self.assertIsNotNone(derived)
            self.assertTrue(cache.has_instrumental("testsha1"))

            # Verify vectorized STFT Wiener mask preserves music and strongly suppresses vocal
            derived_wav = root / "derived_check.wav"
            convert_audio_ffmpeg(derived, derived_wav, sample_rate=sr, channels=2)
            d_arr, _ = sf.read(str(derived_wav), dtype="float32", always_2d=True)
            n_cmp = min(d_arr.shape[0], sr)
            # Check spectral power at 440Hz (vocal) vs 220Hz (left music) and 330Hz (right music)
            spec_l = np.abs(np.fft.rfft(d_arr[:n_cmp, 0]))
            spec_r = np.abs(np.fft.rfft(d_arr[:n_cmp, 1]))
            freqs = np.fft.rfftfreq(n_cmp, d=1.0 / sr)
            band = lambda sp, f0: float(np.max(sp[np.abs(freqs - f0) <= 15.0]))
            self.assertGreater(band(spec_l, 220.0), band(spec_l, 440.0) * 5.0)
            self.assertGreater(band(spec_r, 330.0), band(spec_r, 440.0) * 5.0)

            # 3. API endpoint /api/songscriptions/{sid}/instrumental-audio serves audio cleanly
            store = SongscriptionStore(root)
            item = store.create("Inst Test", src_wav, source_kind="upload")
            sid = item["id"]
            sha1 = item["audio_sha1"]
            sf.write(str(store.cache.vocal_path(sha1)), vocal_sig, sr, subtype="PCM_16")
            self.assertTrue(store.has_instrumental_audio(sha1))
            with patch("cover_studio.modal_client.is_modal_enabled", return_value=False):
                inst_path = store.instrumental_audio_path(sid)
                self.assertIsNotNone(inst_path)
                self.assertTrue(inst_path.is_file())

                app = create_app(root, FakeEngine())
                with TestClient(app) as client:
                    detail = client.get(f"/api/songscriptions/{sid}")
                    self.assertEqual(detail.status_code, 200)
                    self.assertTrue(detail.json()["instrumental_cached"])
                    audio_res = client.get(f"/api/songscriptions/{sid}/instrumental-audio")
                    self.assertEqual(audio_res.status_code, 200)
                    self.assertEqual(audio_res.headers.get("cache-control"), "no-cache")
                    self.assertGreater(len(audio_res.content), 1024)


if __name__ == "__main__":
    unittest.main()

