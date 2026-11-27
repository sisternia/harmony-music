"""Unit tests for Smart MIDI Repair analysis, non-destructive preview, and MIDI export."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from symusic import ControlChange, Note, Score, Tempo, TextMeta, Track

from cover_studio.midi_repair import RepairOptions, preview_midi_repair, write_repaired_midi


def _make_dirty_midi(path: Path) -> Path:
    score = Score(ttype="second")
    score.tempos.append(Tempo(0.0, qpm=120.0, ttype="second"))
    score.markers.append(TextMeta(0.0, "C:maj", "second"))
    score.markers.append(TextMeta(2.0, "G:maj", "second"))
    score.markers.append(TextMeta(4.0, "A:min", "second"))

    track = Track(name="Piano", program=0, ttype="second")
    # 1. Clean melody + bass notes (slightly jittered by 28ms off 0.5s beat)
    track.notes.append(Note(0.528, 0.45, 72, 85, "second"))  # C5 jittered
    track.notes.append(Note(0.5, 0.9, 48, 70, "second"))  # C3 bass

    # 2. Ghost micro-note (15ms, vel 12)
    track.notes.append(Note(1.0, 0.015, 74, 12, "second"))

    # 3. Duplicate same-pitch double trigger (10ms apart)
    track.notes.append(Note(1.5, 0.5, 64, 80, "second"))
    track.notes.append(Note(1.51, 0.48, 64, 65, "second"))

    # 4. Same-pitch overlap extending past next onset
    track.notes.append(Note(2.0, 0.8, 67, 82, "second"))
    track.notes.append(Note(2.4, 0.4, 67, 84, "second"))

    # 5. Impossible right-hand span (> 15 semitones: C4=60 + G5=79 -> 19 semitones)
    track.notes.append(Note(3.0, 0.5, 60, 72, "second"))
    track.notes.append(Note(3.0, 0.5, 79, 90, "second"))

    # 6. Out-of-range note (MIDI 115 > 108)
    track.notes.append(Note(3.6, 0.4, 115, 75, "second"))

    # 7. Muddy overly long pedal across chord changes (0.0 -> 5.2s) + micro bounce
    track.controls.append(ControlChange(0.0, 64, 127, "second"))
    track.controls.append(ControlChange(5.2, 64, 0, "second"))
    track.controls.append(ControlChange(5.5, 64, 127, "second"))
    track.controls.append(ControlChange(5.54, 64, 0, "second"))

    score.tracks.append(track)
    score.dump_midi(str(path))
    return path


class TestSmartMidiRepair(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.midi = _make_dirty_midi(self.dir / "dirty.mid")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_preview_detects_and_repairs_all_artifacts(self) -> None:
        opts = RepairOptions()
        res = preview_midi_repair(
            self.midi,
            opts,
            source_beats=[i * 0.5 for i in range(14)],
            source_chords=[(0.0, "C"), (2.0, "G"), (4.0, "Am")],
        )
        before_counts = res["before"]["issue_counts"]
        self.assertGreaterEqual(before_counts["ghost_notes"], 1)
        self.assertGreaterEqual(before_counts["overlapping_notes"], 2)
        self.assertGreaterEqual(before_counts["out_of_range"], 1)
        self.assertGreaterEqual(before_counts["hand_span"], 1)
        self.assertGreaterEqual(before_counts["pedal_glitches"], 2)

        # After repair: score improves and critical issues drop cleanly
        self.assertGreater(res["after"]["scores"]["overall"], res["before"]["scores"]["overall"])
        self.assertEqual(res["after"]["issue_counts"]["ghost_notes"], 0)
        self.assertEqual(res["after"]["issue_counts"]["out_of_range"], 0)
        self.assertEqual(res["after"]["issue_counts"]["hand_span"], 0)
        self.assertEqual(res["after"]["issue_counts"]["overlapping_notes"], 0)
        self.assertGreater(res["diff"]["actions"]["ghosts_removed"], 0)
        self.assertGreater(res["diff"]["actions"]["pedals_cleaned"], 0)

    def test_manual_delete_and_keep_override(self) -> None:
        opts = RepairOptions()
        first = preview_midi_repair(self.midi, opts)
        ghost_entry = next(n for n in first["diff"]["notes"] if n["status"] == "removed" and "ghost_removed" in n["reasons"])
        kept_entry = next(n for n in first["diff"]["notes"] if n["status"] != "removed")

        res = preview_midi_repair(
            self.midi,
            opts,
            manual_keeps=[ghost_entry["idx"]],
            manual_deletes=[kept_entry["idx"]],
        )
        by_idx = {n["idx"]: n for n in res["diff"]["notes"]}
        self.assertNotEqual(by_idx[ghost_entry["idx"]]["status"], "removed")
        self.assertEqual(by_idx[kept_entry["idx"]]["status"], "manual_deleted")

    def test_write_repaired_midi_roundtrip_preserves_metadata(self) -> None:
        out_path = self.dir / "repaired.mid"
        opts = RepairOptions()
        summary = write_repaired_midi(
            self.midi,
            out_path,
            opts,
            is_source_midi=False,
            metadata_source_path=self.midi,
            source_beats=[i * 0.5 for i in range(14)],
            source_chords=[(0.0, "C"), (2.0, "G"), (4.0, "Am")],
        )
        self.assertTrue(out_path.is_file())
        self.assertGreater(summary["note_count"], 0)

        reloaded = Score(str(out_path)).to("second")
        self.assertGreaterEqual(len(reloaded.tempos), 1)
        self.assertGreaterEqual(len(reloaded.markers), 3)
        pitches = [n.pitch for t in reloaded.tracks for n in t.notes]
        self.assertTrue(all(21 <= p <= 108 for p in pitches))

    def test_api_preview_apply_new_take_overwrite_and_revert(self) -> None:
        from fastapi.testclient import TestClient

        from cover_studio.app import create_app
        from cover_studio.project import Project

        class _DummyEngine:
            tsumugi = None

            def describe(self) -> dict:
                return {"models": [], "score_models": [], "loaded": None, "device": "cpu", "tsumugi": False}

        proj_root = self.dir / "projects"
        proj = Project.create(proj_root, "Test Song", midi=self.midi)
        take = proj.new_takes(
            1,
            {
                "params": {"channel": 227},
                "model": {"id": "m1", "label": "Model 1"},
                "seed": 42,
                "count": 1,
                "continue_from": None,
            },
        )[0]
        import shutil

        shutil.copy2(self.midi, proj.take_midi(take["id"]))
        proj.update_take(take["id"], state="done", notes=10, duration=5.5)

        app = create_app(proj_root, _DummyEngine())
        with TestClient(app) as client:
            # 1. Preview
            r = client.post(
                f"/api/projects/{proj.id}/midi-repair/preview",
                json={"target": "take", "take_id": take["id"], "options": {}},
            )
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertFalse(body["has_backup"])
            self.assertGreater(body["after"]["scores"]["overall"], body["before"]["scores"]["overall"])

            # 2. Apply as new take
            r2 = client.post(
                f"/api/projects/{proj.id}/midi-repair/apply",
                json={"target": "take", "take_id": take["id"], "save_mode": "new_take", "options": {}},
            )
            self.assertEqual(r2.status_code, 200)
            created_id = r2.json()["created_take_id"]
            self.assertTrue(proj.take_midi(created_id).is_file())
            self.assertEqual(len(r2.json()["project"]["takes"]), 2)

            # 3. Overwrite take & verify backup + revert
            r3 = client.post(
                f"/api/projects/{proj.id}/midi-repair/apply",
                json={"target": "take", "take_id": take["id"], "save_mode": "overwrite_take", "options": {}},
            )
            self.assertEqual(r3.status_code, 200)
            self.assertTrue(r3.json()["has_backup"])

            r4 = client.post(
                f"/api/projects/{proj.id}/midi-repair/revert",
                json={"target": "take", "take_id": take["id"]},
            )
            self.assertEqual(r4.status_code, 200)
            self.assertTrue(r4.json()["reverted"])


if __name__ == "__main__":
    unittest.main()
