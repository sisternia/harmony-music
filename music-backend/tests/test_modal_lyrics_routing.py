"""Cloud-only Songscription routing without credentials, model downloads or GPUs."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cover_studio.engine import Cancelled
from cover_studio.lyrics_service import execute_lyrics_pipeline, retranscribe_flagged_segments
from cover_studio.modal_client import ModalConfigurationError, is_modal_enabled


class ModalConfigurationTests(unittest.TestCase):
    def check_config(self, env, module):
        with patch.dict(os.environ, env, clear=True), \
             patch("cover_studio.modal_client._load_backend_dotenv"), \
             patch.dict(sys.modules, {"modal": module}):
            return is_modal_enabled()

    def test_explicit_remote_requires_both_credentials(self):
        for env in (
            {"USE_MODAL_GPU": "true"},
            {"USE_MODAL_GPU": "1", "MODAL_TOKEN_ID": "test-id"},
            {"USE_MODAL_GPU": "yes", "MODAL_TOKEN_SECRET": "test-secret"},
        ):
            with self.subTest(env_keys=list(env)), self.assertRaises(ModalConfigurationError):
                self.check_config(env, SimpleNamespace())

    def test_remote_requires_sdk_in_backend_interpreter(self):
        with self.assertRaisesRegex(ModalConfigurationError, "Modal SDK"):
            self.check_config({
                "USE_MODAL_GPU": "true", "MODAL_TOKEN_ID": "test-id", "MODAL_TOKEN_SECRET": "test-secret",
            }, None)

    def test_configured_remote_and_explicit_local(self):
        self.assertTrue(self.check_config({
            "USE_MODAL_GPU": "true", "MODAL_TOKEN_ID": "test-id", "MODAL_TOKEN_SECRET": "test-secret",
        }, SimpleNamespace()))
        self.assertFalse(self.check_config({"USE_MODAL_GPU": "false"}, None))
        self.assertFalse(self.check_config({}, None))


class ModalLyricsRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "song.wav"
        self.source.write_bytes(b"RIFF-test-only")
        self.cache = MagicMock()
        self.cache.has_vocal.return_value = False
        self.progress = MagicMock()
        self.local_audio = patch("cover_studio.lyrics_service.convert_audio_ffmpeg").start()
        self.local_separation = patch("cover_studio.lyrics_service.isolate_vocals_cached").start()
        self.local_model = patch("cover_studio.lyrics_service.run_music_asr_pipeline").start()
        self.local_refine = patch("cover_studio.lyrics_service.refine_segments_with_music_asr").start()
        patch("cover_studio.lyrics_service.ensure_windows_runtime").start()
        self.addCleanup(patch.stopall)

    def extract(self, **kwargs):
        return execute_lyrics_pipeline(
            engine=SimpleNamespace(unload=lambda: None), cache=self.cache, source_audio=self.source,
            audio_sha1="test-sha", progress=self.progress, should_stop=kwargs.pop("should_stop", lambda: False),
            log=lambda _: None, **kwargs,
        )

    def refine(self):
        return retranscribe_flagged_segments(
            cache=self.cache, source_audio=self.source, audio_sha1="test-sha",
            segments=[{"id": 1, "start": 0., "end": 1., "text": "Ooh", "review_status": "needs_review"}],
        )

    def assert_no_local_models(self):
        self.local_audio.assert_not_called()
        self.local_separation.assert_not_called()
        self.local_model.assert_not_called()
        self.local_refine.assert_not_called()

    def test_remote_extraction_failure_never_falls_back_or_exposes_exception_content(self):
        with patch("cover_studio.modal_client.is_modal_enabled", return_value=True), \
             patch("cover_studio.modal_client.modal_extract_lyrics", side_effect=RuntimeError("private-source-message")):
            with self.assertRaisesRegex(RuntimeError, "Songscription trên Modal") as error:
                self.extract()
        self.assertNotIn("private-source-message", str(error.exception))
        self.assert_no_local_models()
        self.progress.assert_any_call(None, "processing_gpu")

    def test_remote_success_returns_cloud_transcript(self):
        transcript = {"segments": [{"text": "La la la"}], "language_name": "English"}
        with patch("cover_studio.modal_client.is_modal_enabled", return_value=True), \
             patch("cover_studio.modal_client.modal_extract_lyrics", return_value={"transcript": transcript}):
            self.assertIs(self.extract(), transcript)
        self.assert_no_local_models()

    def test_remote_cancellation_before_dispatch_or_after_response_does_not_run_local(self):
        for after_dispatch in (False, True):
            with self.subTest(after_dispatch=after_dispatch), \
                 patch("cover_studio.modal_client.is_modal_enabled", return_value=True), \
                 patch("cover_studio.modal_client.modal_extract_lyrics", return_value={}) as remote:
                stopped = MagicMock(side_effect=[False, True]) if after_dispatch else lambda: True
                with self.assertRaises(Cancelled):
                    self.extract(should_stop=stopped)
                self.assertEqual(remote.call_count, int(after_dispatch))
        self.assert_no_local_models()

    def test_configuration_error_reaches_caller_for_both_operations(self):
        for operation in (self.extract, self.refine):
            with self.subTest(operation=operation.__name__), \
                 patch("cover_studio.modal_client.is_modal_enabled", side_effect=ModalConfigurationError("Missing SDK")):
                with self.assertRaises(ModalConfigurationError):
                    operation()
        self.assert_no_local_models()

    def test_remote_retranscription_failure_and_cancellation_never_fall_back(self):
        for error in (RuntimeError("private-source-message"), Cancelled()):
            with self.subTest(error=type(error).__name__), \
                 patch("cover_studio.modal_client.is_modal_enabled", return_value=True), \
                 patch("cover_studio.modal_client.modal_retranscribe_flagged", side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    self.refine()
                self.assertNotIn("private-source-message", str(caught.exception))
        self.assert_no_local_models()

    def test_explicit_local_mode_still_enters_local_preparation(self):
        self.local_audio.side_effect = ValueError("local-preparation")
        with patch("cover_studio.modal_client.is_modal_enabled", return_value=False), \
             patch("cover_studio.modal_client.modal_extract_lyrics") as remote:
            with self.assertRaisesRegex(ValueError, "local-preparation"):
                self.extract()
        self.local_audio.assert_called_once()
        remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
