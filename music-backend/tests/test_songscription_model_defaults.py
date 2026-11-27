"""Default and legacy Songscription model routing without model downloads or GPU work."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cover_studio.app import LyricsAlignIn, LyricsExtractIn, SongscriptionAlignIn, create_app
from cover_studio.music_asr_engine import DEFAULT_MODEL_ID, QWEN3_ASR_ID
from cover_studio.project import read_json, write_json
from cover_studio.songscription import SongscriptionStore, run_songscription_job


class ModelDefaultsTests(unittest.TestCase):
    def test_request_defaults_select_qwen(self):
        self.assertEqual(DEFAULT_MODEL_ID, QWEN3_ASR_ID)
        self.assertEqual(LyricsExtractIn().model, QWEN3_ASR_ID)
        self.assertEqual(LyricsAlignIn(lyrics_text="La la la").model, QWEN3_ASR_ID)
        self.assertEqual(SongscriptionAlignIn(lyrics_text="La la la").model, QWEN3_ASR_ID)

    def test_upload_and_alignment_migrate_legacy_model_without_running_a_job(self):
        with tempfile.TemporaryDirectory() as tmp, patch("cover_studio.jobs.Runner.submit"):
            engine = type("FakeEngine", (), {"tsumugi": object()})()
            with TestClient(create_app(Path(tmp), engine)) as client:
                choices = client.get("/api/songscriptions").json()["models"]
                self.assertEqual([model["id"] for model in choices], [QWEN3_ASR_ID, "whisper-large-v3"])
                for extra in ({}, {"model": "gemini-songscription"}):
                    with self.subTest(extra=extra):
                        response = client.post(
                            "/api/songscriptions", data=extra,
                            files={"audio": ("song.wav", b"RIFF" + b"\x00" * 200, "audio/wav")},
                        )
                        self.assertEqual(response.status_code, 200, response.text)
                        item = response.json()
                        self.assertEqual(item["model"], QWEN3_ASR_ID)
                        aligned = client.post(
                            f"/api/songscriptions/{item['id']}/align",
                            json={"lyrics_text": "La la la", "model": "gemini-songscription"},
                        )
                        self.assertEqual(aligned.status_code, 200, aligned.text)
                        self.assertEqual(aligned.json()["model"], QWEN3_ASR_ID)

    def test_legacy_saved_session_retranscribes_with_qwen_and_persists_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "song.wav"
            source.write_bytes(b"RIFF" + b"\x00" * 200)
            store = SongscriptionStore(root)
            item = store.create("Song", source, source_kind="upload")
            segments = [{"id": 1, "start": 0., "end": 1., "text": "La la la", "event_type": "vocable"}]
            store.update(item["id"], transcript={"language": "en", "segments": segments}, state="done")
            metadata = store.dir / item["id"] / "item.json"
            write_json(metadata, {**read_json(metadata), "model": "gemini-songscription"})
            with patch("cover_studio.songscription.retranscribe_flagged_segments", return_value=(segments, [])) as transcribe:
                result = store.retranscribe_flagged(item["id"])
            self.assertEqual(transcribe.call_args.kwargs["model_id"], QWEN3_ASR_ID)
            self.assertEqual(result["item"]["model"], QWEN3_ASR_ID)
            self.assertEqual(read_json(metadata)["model"], QWEN3_ASR_ID)

    def test_legacy_saved_job_dispatches_and_records_qwen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "song.wav"
            source.write_bytes(b"RIFF" + b"\x00" * 200)
            store = SongscriptionStore(root)
            item = store.create("Song", source, source_kind="upload")
            metadata = store.dir / item["id"] / "item.json"
            write_json(metadata, {**read_json(metadata), "model": "gemini-songscription"})
            transcript = {
                "model": QWEN3_ASR_ID, "language": "en", "language_name": "English",
                "language_confidence": .8, "detected_languages": [], "vocal_isolated": False,
                "vocal_cached": False, "alignment_method": "line", "duration": 1., "segments": [],
            }
            with patch("cover_studio.songscription.execute_lyrics_pipeline", return_value=transcript) as transcribe:
                run_songscription_job(store, object(), item["id"], None, lambda *_: None, lambda: False, lambda _: None)
            self.assertEqual(transcribe.call_args.kwargs["model_id"], QWEN3_ASR_ID)
            self.assertEqual(store.get(item["id"])["model"], QWEN3_ASR_ID)


if __name__ == "__main__":
    unittest.main()
