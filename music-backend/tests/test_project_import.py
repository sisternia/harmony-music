"""Exercise the file and YouTube paths through the project API."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cover_studio.app import create_app
from cover_studio.youtube import YouTubeImportError, normalize_youtube_url


class FakeEngine:
    tsumugi = object()

    def transcribe(self, *args):
        pass


class ProjectImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app(Path(self.temp.name), FakeEngine())
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_audio_upload_creates_project_and_keeps_audio(self):
        response = self.client.post(
            "/api/projects",
            data={"title": "My song"},
            files={"audio": ("song.mp3", b"example audio", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        project = response.json()
        self.assertEqual(project["title"], "My song")
        self.assertEqual(project["audio"], "song.mp3")
        self.assertEqual((Path(self.temp.name) / project["id"] / "audio" / "song.mp3").read_bytes(), b"example audio")

    def test_midi_upload_creates_project_without_transcription(self):
        response = self.client.post(
            "/api/projects",
            files={"midi": ("song.mid", b"MThd", "audio/midi")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        project = response.json()
        self.assertEqual(project["title"], "song")
        self.assertEqual(project["source"]["origin"], "upload")
        self.assertIsNone(project["transcribe"])

    def test_youtube_link_uses_downloaded_audio_and_video_title(self):
        def fake_download(url: str, directory: Path):
            self.assertEqual(url, "https://youtu.be/dQw4w9WgXcQ")
            audio = directory / "source.mp3"
            audio.write_bytes(b"downloaded audio")
            return audio, "Video title"

        with patch("cover_studio.app.download_youtube_audio", side_effect=fake_download):
            response = self.client.post("/api/projects", data={"youtube": "https://youtu.be/dQw4w9WgXcQ"})
        self.assertEqual(response.status_code, 200, response.text)
        project = response.json()
        self.assertEqual(project["title"], "Video title")
        self.assertEqual((Path(self.temp.name) / project["id"] / "audio" / "source.mp3").read_bytes(), b"downloaded audio")

    def test_invalid_youtube_link_does_not_create_project(self):
        response = self.client.post("/api/projects", data={"youtube": "https://example.com/watch?v=dQw4w9WgXcQ"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/projects").json(), [])

    def test_mixed_inputs_are_rejected(self):
        response = self.client.post(
            "/api/projects",
            data={"youtube": "https://youtu.be/dQw4w9WgXcQ"},
            files={"audio": ("song.mp3", b"audio", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 400)


class YouTubeURLTests(unittest.TestCase):
    def test_video_urls_are_canonicalized(self):
        for value in (
            "https://youtu.be/dQw4w9WgXcQ?t=10",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=ignored",
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_youtube_url(value), "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_non_video_urls_are_rejected(self):
        for value in ("https://example.com/", "https://youtube.com/playlist?list=x", "file:///etc/passwd"):
            with self.subTest(value=value), self.assertRaises(YouTubeImportError):
                normalize_youtube_url(value)


if __name__ == "__main__":
    unittest.main()
