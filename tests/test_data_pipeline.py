from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from looktwice.data.extract import ExtractConfig, process_video, run_extract, sample_indices
from looktwice.data.faces import Detection, square_crop
from looktwice.data.metadata import FAKE, REAL, label_from_path, parse_name, scan_videos
from looktwice.data.split import apply_split, assign_actors, list_actors, search_seeds


@pytest.mark.parametrize("stem,expected", [
    ("01__exit_phone_room", {"target": "01", "donor": None, "scene": "exit_phone_room", "uid": None}),
    ("01_02__exit_phone_room__YVGY8LOK", {"target": "01", "donor": "02", "scene": "exit_phone_room", "uid": "YVGY8LOK"}),
    ("28_15__walking_down_indoor_hall_disgust__ABC123", {"target": "28", "donor": "15",
                                                         "scene": "walking_down_indoor_hall_disgust", "uid": "ABC123"}),
    ("1_3__talking", {"target": "01", "donor": "03", "scene": "talking", "uid": None}),
])
def test_parse_name(stem, expected):
    assert parse_name(stem) == expected


@pytest.mark.parametrize("stem", ["random_video", "__scene", "ab__scene"])
def test_parse_name_rejects(stem):
    assert parse_name(stem) is None


def test_label_from_path():
    assert label_from_path(Path("DFD_original sequences/01__a.mp4")) == REAL
    assert label_from_path(Path("DFD_manipulated_sequences/x/01_02__a__X.mp4")) == FAKE
    assert label_from_path(Path("other/01__a.mp4")) is None


def write_video(path: Path, n_frames=40, size=(160, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for i in range(n_frames):
        vw.write(np.full((size[1], size[0], 3), i * 5 % 255, np.uint8))
    vw.release()


def make_dataset(root: Path, n_actors=8):
    actors = [f"{i:02d}" for i in range(1, n_actors + 1)]
    for a in actors:
        write_video(root / "DFD_original sequences" / f"{a}__scene_a.mp4")
        for d in actors:
            if d != a:
                write_video(root / "DFD_manipulated_sequences" / f"{a}_{d}__scene_a__U{a}{d}.mp4", n_frames=20)
    return actors


class DummyDetector:
    def detect(self, img):
        h, w = img.shape[:2]
        return [Detection(w * 0.3, h * 0.25, w * 0.4, h * 0.5, 0.99)]


class NoFaceDetector:
    def detect(self, img):
        return []


def test_scan_videos(tmp_path):
    actors = make_dataset(tmp_path, n_actors=4)
    (tmp_path / "DFD_original sequences" / "weird_name.mp4").write_bytes(b"")
    scan = scan_videos(tmp_path)
    assert (scan.videos.label == REAL).sum() == 4
    assert (scan.videos.label == FAKE).sum() == 4 * 3
    assert scan.unparsed == ["DFD_original sequences/weird_name.mp4"]
    assert list_actors(scan.videos) == actors


def test_split_has_no_actor_leakage(tmp_path):
    make_dataset(tmp_path, n_actors=8)
    videos = scan_videos(tmp_path).videos
    actor_split = assign_actors(list_actors(videos), n_val=2, n_test=2, seed=0)
    assert sorted(actor_split.values()).count("test") == 2

    target = apply_split(videos, actor_split, "target")
    assert (target.split != "drop").all()
    for split in ("train", "val", "test"):
        assert set(target[target.split == split].target) <= {a for a, s in actor_split.items() if s == split}

    strict = apply_split(videos, actor_split, "strict")
    kept_fakes = strict[(strict.label == FAKE) & (strict.split != "drop")]
    assert (kept_fakes.donor.map(actor_split) == kept_fakes.split).all()
    assert (strict.split == "drop").sum() > 0


def test_assign_actors_is_deterministic():
    actors = [f"{i:02d}" for i in range(28)]
    assert assign_actors(actors, 4, 4, 7) == assign_actors(actors, 4, 4, 7)
    with pytest.raises(ValueError):
        assign_actors(actors[:5], 3, 2, 0)


def test_search_seeds(tmp_path):
    make_dataset(tmp_path, n_actors=6)
    videos = scan_videos(tmp_path).videos
    res = search_seeds(videos, 1, 1, n_seeds=5)
    assert len(res) == 5 and res.min_val_test.is_monotonic_decreasing


def test_sample_indices():
    idx = sample_indices(100, 4, margin=0.05)
    assert len(idx) == 4 and all(5 <= i < 95 for i in idx) and idx == sorted(idx)
    assert sample_indices(3, 10) == [0, 1, 2]
    assert sample_indices(0, 5) == []


def test_square_crop_is_square_and_padded():
    img = np.full((100, 200, 3), 255, np.uint8)
    crop = square_crop(img, Detection(-10, -10, 40, 60, 1.0), scale=1.2, out_size=64)
    assert crop.shape == (64, 64, 3)
    assert crop[0, 0].sum() == 0
    assert crop[-1, -1].sum() > 0


def test_process_video_and_run_extract(tmp_path, monkeypatch):
    root, out = tmp_path / "dfd", tmp_path / "out"
    make_dataset(root, n_actors=4)
    videos = scan_videos(root).videos
    videos = apply_split(videos, assign_actors(list_actors(videos), 1, 1, 0), "target")
    cfg = ExtractConfig(frames_real=5, frames_fake=2, save_size=32)

    row = videos[videos.label == REAL].iloc[0].to_dict()
    saved, failed = process_video((row, str(root), str(out), cfg), detector=DummyDetector())
    assert len(saved) == 5 and not failed
    img = cv2.imread(str(out / saved[0]["path"]))
    assert img.shape == (32, 32, 3)

    saved, failed = process_video((row, str(root), str(out), cfg), detector=NoFaceDetector())
    assert not saved and len(failed) == 5

    import looktwice.data.extract as ex
    monkeypatch.setattr(ex, "build_detector", lambda *a, **k: DummyDetector())
    index = run_extract(videos, root, out, cfg, workers=1)
    n_real, n_fake = (videos.label == REAL).sum(), (videos.label == FAKE).sum()
    assert len(index) == n_real * 5 + n_fake * 2
    assert set(index.columns) >= {"path", "label", "split", "video_id", "target", "donor"}
    assert pd.read_csv(out / "crops_index.csv").shape[0] == len(index)
    assert all((out / p).exists() for p in index.path)
