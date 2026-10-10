from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from courtship_movie import CourtshipMovie, CourtshipMovieError, N_COLUMNS  # noqa: E402


def _directions(x: float = 1.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    direction = np.asarray([x, y, z], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    return np.broadcast_to(direction, (N_COLUMNS, 3)).copy()


class CourtshipMovieTests(unittest.TestCase):
    def test_monocular_mode_replicates_one_exact_vector_and_saves_narrow_artifact(self) -> None:
        left_dirs = _directions()
        # The right geometry is deliberately different; mono mode must not
        # manufacture a second view from it.
        right_dirs = _directions(0.0, 1.0, 0.0)
        frames = np.stack((np.full((4, 8), 32, np.uint8),
                           np.full((4, 8), 224, np.uint8)))
        stimulus = CourtshipMovie.from_frames(
            frames,
            left_directions=left_dirs,
            right_directions=right_dirs,
            eye_mode="monocular_replicated",
        )

        left0, right0 = stimulus.sample(0.0)
        left1, right1 = stimulus.sample(0.005)
        self.assertEqual(left0.shape, (721,))
        self.assertEqual(right0.shape, (721,))
        np.testing.assert_array_equal(left0, right0)
        np.testing.assert_array_equal(left1, right1)
        self.assertAlmostEqual(float(left0[0]), 32 / 255, places=6)
        self.assertAlmostEqual(float(left1[0]), 224 / 255, places=6)
        manifest = stimulus.manifest()
        self.assertEqual(manifest["eye_mode"], "monocular_replicated")
        self.assertEqual(manifest["right_eye_sampling"], "copy of left vector; not binocular POV")
        self.assertEqual(manifest["sample_dt_s"], 0.005)

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "courtship-columns.npz"
            stimulus.save(artifact)
            loaded = CourtshipMovie.from_artifact(
                artifact, left_directions=left_dirs, right_directions=right_dirs,
            )
            np.testing.assert_array_equal(loaded.sample(0.005)[0], left1)
            with self.assertRaises(CourtshipMovieError):
                CourtshipMovie.from_artifact(
                    artifact, left_directions=right_dirs, right_directions=right_dirs,
                )

    def test_binocular_side_by_side_uses_separate_halves_and_eye_geometries(self) -> None:
        # Each 2x4 half has a constant but distinct luminance. Keeping the
        # matching directions forward maps all columns into each half.
        frame = np.concatenate((np.full((3, 4), 51, np.uint8),
                                np.full((3, 4), 204, np.uint8)), axis=1)
        stimulus = CourtshipMovie.from_frames(
            frame[None, ...],
            left_directions=_directions(),
            right_directions=_directions(0.0, 1.0, 0.0),
            eye_mode="binocular_pov",
            stereo_layout="side_by_side_lr",
        )
        left, right = stimulus.sample(0.0)
        self.assertAlmostEqual(float(left.mean()), 51 / 255, places=6)
        self.assertAlmostEqual(float(right.mean()), 204 / 255, places=6)
        self.assertNotEqual(stimulus.manifest()["eye_mode"], "monocular_replicated")

    def test_equirectangular_projection_samples_full_direction_range(self) -> None:
        height, width = 9, 36
        # Horizontal gradient lets the test check wrapping and values without
        # relying on a fixed pixel-space pseudo-eye grid.
        frame = np.broadcast_to(np.arange(width, dtype=np.uint8), (height, width))
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        left = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        directions = np.stack((forward, left), axis=0)
        directions = np.pad(directions, ((0, N_COLUMNS - 2), (0, 0)))
        directions[2:] = forward
        stimulus = CourtshipMovie.from_frames(
            frame[None, ...], left_directions=directions,
            right_directions=directions, eye_mode="monocular_replicated",
            projection="equirectangular",
        )
        left_columns, _ = stimulus.sample(0.0)
        self.assertLess(left_columns[0], 0.1)
        self.assertLess(left_columns[1], 0.1)
        self.assertEqual(stimulus.manifest()["out_of_fov_sample_count"]["left"], 0)

    def test_geometry_shape_values_sampling_grid_and_mode_are_strict(self) -> None:
        frames = np.zeros((1, 4, 4), np.uint8)
        with self.assertRaises(CourtshipMovieError):
            CourtshipMovie.from_frames(
                frames, left_directions=np.zeros((720, 3)),
                right_directions=_directions(), eye_mode="monocular_replicated",
            )
        invalid = _directions()
        invalid[3, 0] = np.nan
        with self.assertRaises(CourtshipMovieError):
            CourtshipMovie.from_frames(
                frames, left_directions=invalid, right_directions=_directions(),
                eye_mode="monocular_replicated",
            )
        with self.assertRaises(CourtshipMovieError):
            CourtshipMovie.from_frames(
                frames, left_directions=_directions(), right_directions=_directions(),
                eye_mode="binocular_pov",
            )
        with self.assertRaises(CourtshipMovieError):
            CourtshipMovie.from_frames(
                frames, left_directions=_directions(), right_directions=_directions(),
                eye_mode="monocular_replicated", stereo_layout="side_by_side_lr",
            )
        stimulus = CourtshipMovie.from_frames(
            frames, left_directions=_directions(), right_directions=_directions(),
            eye_mode="monocular_replicated",
        )
        with self.assertRaises(CourtshipMovieError):
            stimulus.sample(0.0025)
        with self.assertRaises(IndexError):
            stimulus.sample(0.005)

    def test_exact_rendered_columns_round_trip_with_provenance_and_geometry_hashes(self) -> None:
        left_dirs, right_dirs = _directions(), _directions(0.0, 1.0, 0.0)
        columns = np.zeros((3, 2, N_COLUMNS), dtype=np.float32)
        columns[:, 0] = 0.25
        columns[:, 1] = 0.75
        stimulus = CourtshipMovie.from_exact_columns(
            columns,
            left_directions=left_dirs,
            right_directions=right_dirs,
            source_metadata={
                "sample_hz": 200,
                "render_backend": "warp_cuda",
                "pose_mode": "fixed",
                "scene_sha256": "a" * 64,
            },
        )
        manifest = stimulus.manifest()
        self.assertEqual(manifest["eye_mode"], "binocular_world_render")
        self.assertEqual(manifest["projection"], "direct_exact_columns")
        self.assertEqual(manifest["source_metadata"]["pose_mode"], "fixed")
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "fixed-pose-columns.npz"
            stimulus.save(artifact)
            loaded = CourtshipMovie.from_artifact(
                artifact, left_directions=left_dirs, right_directions=right_dirs,
            )
            self.assertEqual(loaded.manifest()["eye_mode"], "binocular_world_render")
            np.testing.assert_array_equal(loaded.sample(0.01)[0], columns[2, 0])
            with self.assertRaises(CourtshipMovieError):
                CourtshipMovie.from_exact_columns(
                    columns[:, :, :720], left_directions=left_dirs,
                    right_directions=right_dirs, source_metadata={"sample_hz": 200},
                )
            with self.assertRaises(CourtshipMovieError):
                CourtshipMovie.from_exact_columns(
                    columns, left_directions=left_dirs, right_directions=right_dirs,
                    source_metadata={"sample_hz": 100},
                )

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "FFmpeg integration tools are unavailable")
    def test_from_video_decodes_at_exact_200_hz_and_hashes_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "gray.mp4"
            make_video = subprocess.run(
                [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-f", "lavfi",
                 "-i", "color=c=gray:s=8x4:r=10:d=0.2", "-c:v", "mpeg4",
                 "-pix_fmt", "yuv420p", "-y", str(video)],
                check=False, capture_output=True, text=True,
            )
            if make_video.returncode != 0:
                self.skipTest(f"FFmpeg cannot make a small local test movie: {make_video.stderr}")
            stimulus = CourtshipMovie.from_video(
                video, left_directions=_directions(), right_directions=_directions(),
                eye_mode="monocular_replicated",
            )
            self.assertEqual(stimulus.n_frames, 40)
            self.assertEqual(stimulus.manifest()["decoder"], "ffmpeg/ffprobe")
            self.assertEqual(len(stimulus.manifest()["source_sha256"]), 64)
            self.assertEqual(stimulus.sample(0.195)[0].shape, (721,))


if __name__ == "__main__":
    unittest.main()
