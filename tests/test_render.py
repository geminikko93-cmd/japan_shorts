"""편집·오디오 회귀 테스트 (외부 API/다운로드 없음, ffmpeg 필요).

합성 소스 영상은 프레임 번호를 화면 전체 색으로 인코딩한다 → 출력 프레임을 디코딩해
'선택 구간 밖 프레임이 나왔는지'를 프레임 단위로 확인할 수 있다.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shorts import composer, director, pipeline, sources  # noqa: E402
from shorts.common import PipelineError, validate_config  # noqa: E402
from shorts.layout import crop_for, get_layout  # noqa: E402
from shorts.plan import (BOUND_EPS, LEAD_SEC, apply_caption_edits, apply_cut_edits,  # noqa: E402
                         apply_marker_proposals, audio_status, auto_plan, caption_rows, check_plan, cut_rows,
                         load_plan, new_plan, relayout, repetition_warnings, save_plan, validate_for_render)
from shorts.scenes import Clip  # noqa: E402
from shorts.sources import Source  # noqa: E402
from shorts.styles import STYLES, get_style, styled_cfg  # noqa: E402
from shorts.textrender import TextFitError, fit_subtitle, fit_title, render_subtitle, render_title  # noqa: E402

W, H = 640, 360


def base_cfg(tmp: Path) -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["_root"] = str(ROOT)
    cfg["output_dir"] = str(tmp / "output")
    cfg["work_dir"] = str(tmp / "work")
    cfg["video"]["target_duration"] = 4.0
    cfg["video"]["max_duration"] = 5.0
    return cfg


def color(idx: int) -> tuple[int, int, int]:
    return (idx % 16) * 16 + 8, (idx // 16) % 16 * 16 + 8, (idx // 256) % 16 * 16 + 8


def decode_idx(rgb) -> int:
    r, g, b = (int(np.clip(round((float(v) - 8) / 16), 0, 15)) for v in rgb)
    return r + 16 * g + 256 * b


def make_video(path: Path, seconds: float, fps: str = "30") -> Path:
    num, den = (int(x) for x in fps.split("/")) if "/" in fps else (int(fps), 1)
    n = int(round(seconds * num / den))
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", fps, "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
                             "-qp", "0", "-pix_fmt", "yuv444p", "-g", "15", str(path)], stdin=subprocess.PIPE)
    frame = np.empty((H, W, 3), np.uint8)
    for i in range(n):
        frame[:] = color(i)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def frame_indices(video: Path, x: int, y: int, w: int, h: int) -> list[int]:
    """영상의 (x,y,w,h) 영역 평균색 → 프레임 번호 목록."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                          f"crop={w}:{h}:{x}:{y},scale=4:4:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    arr = np.frombuffer(out, np.uint8).reshape(-1, 4, 4, 3)
    return [decode_idx(a.reshape(-1, 3).mean(0)) for a in arr]


def caption_pixels(video: Path, lay) -> list[int]:
    """프레임별 자막 영역의 밝은 픽셀 수 (자막이 있으면 큼)."""
    h = lay.caption_h // 10
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                          f"crop={lay.width}:{lay.caption_h}:0:{lay.caption_y},scale=108:{h},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    arr = np.frombuffer(out, np.uint8).reshape(-1, h, 108)
    return [int((a > 128).sum()) for a in arr]


def audio_samples(video: Path) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a:0", "-f", "s16le",
                          "-ac", "2", "-ar", "44100", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).reshape(-1, 2)


def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,duration,nb_frames",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def clip(cid, src: Path, start, end, fps=30.0, dh=None, face=None) -> Clip:
    return Clip(cid, Source(path=str(src)), start, end, 0.2, face, (W, H), fps,
                dh if dh is not None else hash(cid) & (2 ** 64 - 1), "")


SCRIPT = {"name_ja": "テスト", "name_ko": "테스트", "titles": [{"line1": "テスト", "line2": "この笑顔が好き", "ko": "t"}],
          "captions": [], "hashtags": ["#shorts"]}


def caps(*texts):
    return [{"id": f"c{i + 1:02d}", "ja": t, "ko": ""} for i, t in enumerate(texts)]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg 필요")
class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="shorts_test_"))
        cls.v30 = make_video(cls.tmp / "a30.mp4", 12, "30")
        cls.v6 = make_video(cls.tmp / "b6.mp4", 12, "6")
        cls.v60 = make_video(cls.tmp / "c5994.mp4", 12, "60000/1001")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.cfg = base_cfg(self.tmp)
        self.lay = get_layout(self.cfg)

    def _plan(self, clips, captions, cfg=None, style="praise_list", **kw):
        cfg = cfg or self.cfg
        lay = get_layout(cfg)
        if captions and isinstance(captions[0], str):
            captions = caps(*captions)
        cuts, events, notes = auto_plan(clips, captions, cfg, lay, style=style, **kw)
        srcs = list({c.source.path: c.source for c in clips}.values())
        run = pipeline.new_run_dir(cfg, SCRIPT)
        return new_plan("テスト", 0, ("テスト", "この笑顔が好き"), lay, cfg, style, cuts, events, notes, srcs, run,
                        kw.get("first_clip", ""))

    def _render(self, plan, cfg=None):
        return pipeline.render_plan(cfg or self.cfg, dict(SCRIPT), plan)[0]

    def _assert_mp4(self, out: Path, plan):
        p = probe(out)
        self.assertEqual((p["video"]["width"], p["video"]["height"]), (1080, 1920))
        self.assertEqual(p["video"]["r_frame_rate"], "30/1")
        self.assertEqual(int(p["video"]["nb_frames"]), plan.total_frames)
        vd, ad = float(p["video"]["duration"]), float(p["audio"]["duration"])
        self.assertAlmostEqual(vd, plan.total, delta=1 / 30)
        # AAC는 1024샘플 프레임 단위 → 최대 1프레임(23ms) 차이 허용
        self.assertLessEqual(abs(vd - ad), 1024 / 44100 + 1e-3, f"video {vd} audio {ad}")
        n = len(audio_samples(out))
        self.assertLessEqual(abs(n / 44100 - plan.total), 1024 / 44100 + 1e-3)
        return p

    def three(self):
        return [clip("A", self.v30, 0.0, 3.9), clip("B", self.v30, 4.0, 7.9), clip("C", self.v30, 8.0, 11.9)]

    # ───────────── P0: 효과음 제거 ─────────────
    def test_p0_default_silent_multi_cut(self):
        plan = self._plan(self.three(), ["一つ目のテロップ", "二つ目", "三つ目のテロップです"])
        self.assertGreater(len(plan.cuts), 1)
        out = self._render(plan)
        self._assert_mp4(out, plan)
        self.assertEqual(int(np.abs(audio_samples(out)).max()), 0, "기본 출력에 소리가 있음")
        self.assertFalse(list(Path(self.cfg["work_dir"]).rglob("*.wav")), "효과음 파일이 생성됨")
        self.assertFalse((ROOT / "assets" / "sfx").exists(), "기본 효과음 폴더가 다시 생김")
        self.assertFalse((ROOT / "shorts" / "sfx.py").exists())

    def test_p0_legacy_sfx_keys_ignored_single_cut(self):
        cfg = copy.deepcopy(self.cfg)
        legacy_dir = self.tmp / "legacy_sfx_dir"
        cfg["audio"].update({"sfx_dir": str(legacy_dir), "cut_sfx": "click", "sfx_volume": 0.6})
        cfg["video"].update({"target_duration": 1.5, "max_duration": 2.0})
        plan = self._plan([clip("A", self.v30, 0.0, 3.9)], ["一つだけ"], cfg)
        self.assertEqual(len(plan.cuts), 1)
        out = self._render(plan, cfg)
        self._assert_mp4(out, plan)
        self.assertEqual(int(np.abs(audio_samples(out)).max()), 0)
        self.assertFalse(legacy_dir.exists(), "구버전 sfx_dir가 생성됨")

    def test_p0_every_style_silent_even_with_sfx_override(self):
        for name in STYLES:
            cfg = copy.deepcopy(self.cfg)
            cfg["styles"] = {name: {"sfx": {"cut": "click"}}}          # 스타일 설정에 효과음 키를 넣어도 무시
            cfg["audio"].update({"cut_sfx": "click", "sfx_volume": 1.0})
            self.assertNotIn("sfx", get_style(cfg, name))
            plan = self._plan(self.three(), ["一つ目のテロップ", "二つ目のテロップ"], cfg, style=name)
            out = self._render(plan, cfg)
            self._assert_mp4(out, plan)
            self.assertEqual(int(np.abs(audio_samples(out)).max()), 0, name)

    def test_p0_short_bgm_padded_with_silence(self):
        music = self.tmp / "bgm.wav"
        t = np.arange(int(44100 * 1.0)) / 44100
        pcm = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
        with wave.open(str(music), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(44100), w.writeframes(pcm.tobytes())
        cfg = copy.deepcopy(self.cfg)
        cfg["audio"].update({"music_path": str(music), "music_start": 0.0, "sfx_dir": "assets/sfx", "cut_sfx": "click"})
        plan = self._plan(self.three(), ["一つ目のテロップ", "二つ目のテロップ", "三つ目"], cfg)
        out = self._render(plan, cfg)
        self._assert_mp4(out, plan)
        a = np.abs(audio_samples(out).astype(np.int32))
        self.assertGreater(a[int(0.3 * 44100):int(0.8 * 44100)].max(), 3000, "BGM이 안 들어감")
        self.assertEqual(int(a[int(1.1 * 44100):].max()), 0, "BGM 뒤 구간에 소리(효과음 등)가 있음")
        self.assertGreater(plan.total, 3.0)

    def test_p0_missing_bgm_file_is_error(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["audio"]["music_path"] = str(self.tmp / "nope.mp3")
        plan = self._plan([clip("A", self.v30, 0.0, 3.9)], ["テスト"], cfg)
        with self.assertRaisesRegex(PipelineError, "음원 파일이 없습니다"):
            self._render(plan, cfg)

    def test_music_timing_unverified_by_default(self):
        plan = self._plan(self.three(), ["一つ目"])
        self.assertEqual(plan.audio["timing"], "unverified")
        self.assertIsNone(plan.audio["offset"])           # 곡 제목만으로 후렴 시작을 추측하지 않음
        self.assertIn("미확인", audio_status(plan.audio))

    # ───────────── P1-A: 경계 / 반복 ─────────────
    def _check_segment_frames(self, src: Path, src_fps: float, start: float, end: float):
        c = clip("X", src, start, end, fps=src_fps)
        plan = self._plan([c], ["テスト"])
        s = plan.cuts[0]
        self.assertGreaterEqual(s.src_start, c.start + LEAD_SEC - 1e-3)
        self.assertLessEqual(s.src_start + s.frames / 30, c.end + BOUND_EPS)
        seg = self.tmp / f"seg_{src.stem}_{start}.mp4"
        composer._render_segment(s, seg, self.lay, 30)
        idx = frame_indices(seg, 0, 0, 1080, self.lay.video_h)
        self.assertEqual(len(idx), s.frames)
        lo = s.src_start * src_fps - 0.02
        hi = (s.src_start + s.frames / 30) * src_fps
        bad = [i for i in idx if not (lo <= i < hi)]
        self.assertFalse(bad, f"구간 밖 프레임 {bad[:5]} (허용 {lo:.2f}~{hi:.2f}, fps {src_fps})")
        self.assertEqual(idx, sorted(idx))
        self.assertGreaterEqual(idx[0] / src_fps, s.src_start - 1e-3)
        self.assertLess(idx[0] / src_fps, s.src_start + 1 / 30)
        return idx

    def test_p1_segment_frames_inside_bounds_30fps(self):
        idx = self._check_segment_frames(self.v30, 30.0, 2.0, 3.8)
        self.assertEqual(idx, list(range(idx[0], idx[0] + len(idx))), "30fps는 프레임이 연속이어야 함")

    def test_p1_segment_frames_inside_bounds_6fps(self):
        self._check_segment_frames(self.v6, 6.0, 1.0, 3.0)

    def test_p1_segment_frames_inside_bounds_5994(self):
        self._check_segment_frames(self.v60, 60000 / 1001, 3.0, 4.9)

    def test_p1_segment_at_file_end(self):
        self._check_segment_frames(self.v30, 30.0, 10.0, 12 - 2 / 30)

    def test_p1_offset_inside_scene_stays_in_bounds(self):
        # 같은 장면의 뒤쪽 구간(재사용 오프셋에 해당)을 써도 장면 끝을 넘지 않음
        c = clip("A", self.v30, 0.0, 6.0)
        plan = self._plan([c], ["テスト"])
        rows = cut_rows(plan)
        rows[0].update(start=4.5, seconds=1.4)
        plan = apply_cut_edits(plan, rows, {"A": c}, self.cfg)
        self.assertEqual(check_plan(plan, self.cfg, self.lay)[0], [])
        rows[0].update(start=4.5, seconds=1.6)                      # 6.1초 → 장면 밖
        plan = apply_cut_edits(plan, rows, {"A": c}, self.cfg)
        self.assertTrue(any("벗어남" in e for e in check_plan(plan, self.cfg, self.lay)[0]))

    def test_p1_short_candidates_skipped_and_short_result(self):
        clips = [clip("short", self.v30, 0.0, 1.2), clip("ok", self.v30, 2.0, 4.0)]
        plan = self._plan(clips, ["一つ目のテロップ", "二つ目のテロップ", "三つ目"])
        self.assertEqual([s.clip_id for s in plan.cuts], ["ok"])
        self.assertTrue(any("짧게 완성" in n for n in plan.notes), plan.notes)

    def test_p1_no_reuse_no_overlap_and_max_duration(self):
        clips = [clip(f"c{k}", self.v30, k * 2.0, k * 2.0 + 1.95) for k in range(6)]
        for style in STYLES:
            plan = self._plan(clips, [f"テロップ{k}" for k in range(20)], style=style)
            ids = [s.clip_id for s in plan.cuts]
            self.assertEqual(len(ids), len(set(ids)), "같은 장면 재사용")
            self.assertLessEqual(plan.total, self.cfg["video"]["max_duration"] + 1e-9)
            self.assertEqual(check_plan(plan, self.cfg, self.lay)[0], [], style)

    def test_p1_invalid_inputs(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["video"].update({"target_duration": 10, "max_duration": 5})
        with self.assertRaisesRegex(PipelineError, "max_duration"):
            self._plan([clip("A", self.v30, 0, 3)], ["テスト"], cfg)
        with self.assertRaisesRegex(PipelineError, "자막"):
            self._plan([clip("A", self.v30, 0, 3)], ["", "  "])
        with self.assertRaisesRegex(PipelineError, "장면"):
            self._plan([], ["テスト"])
        with self.assertRaisesRegex(PipelineError, "컷을 하나도"):
            self._plan([clip("A", self.v30, 0, 1.0)], ["テスト"])

    def test_p1_visual_duplicates_vs_same_source(self):
        clips = [clip("A", self.v30, 0, 3.9, dh=0), clip("B", self.v30, 4, 7.9, dh=(2 ** 64 - 1)),
                 clip("A2", self.v60, 0, 3.9, dh=1)]
        plan = self._plan(clips, ["一つ目のテロップ", "二つ目のテロップ", "三つ目のテロップ"])
        self.assertEqual([s.clip_id for s in plan.cuts], ["A", "B", "A2"])
        self.assertTrue(plan.cuts[2].relaxed)
        self.assertTrue(any("완화" in n for n in plan.notes))
        cfg = copy.deepcopy(self.cfg)
        cfg["timeline"]["allow_similar_fallback"] = False
        plan2 = self._plan(clips, ["一つ目のテロップ", "二つ目のテロップ", "三つ目のテロップ"], cfg)
        self.assertEqual([s.clip_id for s in plan2.cuts], ["A", "B"])

    def test_p1_same_person_different_scene_not_duplicate(self):
        feat = np.ones(128) / np.sqrt(128)
        a, b = clip("A", self.v30, 0, 3.9, dh=0), clip("B", self.v30, 4, 7.9, dh=2 ** 64 - 1)
        a.feat = b.feat = feat
        plan = self._plan([a, b], ["一つ目のテロップ", "二つ目のテロップ"])
        self.assertEqual([s.clip_id for s in plan.cuts], ["A", "B"])
        self.assertFalse(any(s.relaxed for s in plan.cuts))

    # ───────────── 청춘·설렘: 컷/자막 독립 타임라인 ─────────────
    def youth_cfg(self, target=10.0):
        cfg = copy.deepcopy(self.cfg)
        cfg["video"].update({"target_duration": target, "max_duration": target + 2})
        return cfg

    def test_youth_caption_spans_multiple_cuts(self):
        cfg = self.youth_cfg(10.0)
        plan = self._plan(self.three() + [clip("D", self.v60, 0, 3.9)], ["この一瞬が好き", "何回でも見たくなる"], cfg,
                          style="youth_romance")
        self.assertGreaterEqual(len(plan.cuts), 3)
        self.assertEqual(len(plan.captions), 2)
        self.assertTrue(any(c.last_cut > c.first_cut for c in plan.captions), "한 자막이 여러 컷에 걸쳐야 함")
        self.assertEqual(plan.cuts[0].role, "key")
        self.assertEqual(check_plan(plan, cfg, get_layout(cfg))[0], [])

    def test_youth_caption_only_edit_keeps_cuts_and_uncaptioned_key_cut(self):
        cfg = self.youth_cfg(10.0)
        clips = self.three() + [clip("D", self.v60, 0, 3.9)]
        plan = self._plan(clips, ["この一瞬が好き", "何回でも見たくなる"], cfg, style="youth_romance")
        cuts_before = [(c.clip_id, c.src_start, c.frames) for c in plan.cuts]
        self.assertGreaterEqual(plan.cuts[0].frames / 30, 3.0)
        rows = caption_rows(plan)
        rows[0].update(text="この笑顔、ずるい", first=2, last=2)       # 1번 컷(3초 이상)은 자막 없이
        rows[1].update(first=3, last=len(plan.cuts))
        plan = apply_caption_edits(plan, rows)
        self.assertEqual(cuts_before, [(c.clip_id, c.src_start, c.frames) for c in plan.cuts], "컷이 바뀜")
        self.assertEqual(check_plan(plan, cfg, get_layout(cfg))[0], [])
        out = self._render(plan, cfg)
        self._assert_mp4(out, plan)
        lay = get_layout(cfg)
        px = caption_pixels(out, lay)
        f1 = plan.cuts[0].frames
        self.assertEqual(max(px[:f1]), 0, "무자막이어야 할 첫 컷에 자막이 있음")
        self.assertGreater(min(px[f1 + 1:plan.cut_starts()[2] - 1]), 50, "2번 컷 자막이 없음")
        self.assertEqual(int(np.abs(audio_samples(out)).max()), 0)

    def test_youth_marker_keeps_context_inside_scene(self):
        cfg = self.youth_cfg(6.0)
        c = clip("A", self.v30, 0.0, 11.9)
        b = clip("B", self.v60, 0, 3.9)
        plan = self._plan([c, b], ["この一瞬が好き"], cfg, style="youth_romance")
        rows = cut_rows(plan)
        rows[0]["marker"] = 6.0                                     # 장면 안 6초에 웃는 순간 (사람이 지정)
        plan = apply_cut_edits(plan, rows, {"A": c, "B": b}, cfg)
        apply_marker_proposals(plan, {"A": c, "B": b}, cfg)
        k = plan.cuts[0]
        st = get_style(cfg, "youth_romance")["cut"]
        self.assertAlmostEqual(k.src_start, 6.0 - st["marker_before_sec"], delta=1 / 30)
        self.assertAlmostEqual(k.src_start + k.frames / 30, 6.0 + st["marker_after_sec"], delta=1 / 30)
        self.assertEqual(k.role, "key")
        rows = cut_rows(plan)
        rows[0]["marker"] = 11.0                                    # 장면 끝 근처: 장면 밖으로 늘리지 않음
        plan = apply_cut_edits(plan, rows, {"A": c, "B": b}, cfg)
        apply_marker_proposals(plan, {"A": c, "B": b}, cfg)
        self.assertLessEqual(plan.cuts[0].src_start + plan.cuts[0].frames / 30, c.end + BOUND_EPS)

    def test_long_continuous_scene_windows(self):
        # 12초짜리 연속 장면 하나 → 후보 구간 2개 (장면 경계 0~11.9 유지)
        wins = [Clip("L-a", Source(path=str(self.v30)), 0.0, 6.0, 0.2, None, (W, H), 30.0, 0, "", None, 0.0, 11.9),
                Clip("L-b", Source(path=str(self.v30)), 6.0, 11.9, 0.2, None, (W, H), 30.0, 2 ** 64 - 1, "", None, 0.0, 11.9)]
        cfg = self.youth_cfg(6.0)
        plan = self._plan(wins, ["この一瞬が好き"], cfg, style="youth_romance")
        self.assertEqual([c.clip_id for c in plan.cuts], ["L-a", "L-b"])
        self.assertEqual(check_plan(plan, cfg, get_layout(cfg))[0], [])
        for c in plan.cuts:
            self.assertEqual((c.scene_start, c.scene_end), (0.0, 11.9))
        # 후보 구간(0~6초) 밖이라도 같은 장면 안이면 늘릴 수 있고, 장면 끝은 넘을 수 없음
        rows = cut_rows(plan)
        rows[1].update(start=7.0, seconds=4.5)
        plan = apply_cut_edits(plan, rows, {c.id: c for c in wins}, cfg)
        self.assertEqual(check_plan(plan, cfg, get_layout(cfg))[0], [])
        rows[1].update(start=7.0, seconds=5.0)
        plan = apply_cut_edits(plan, rows, {c.id: c for c in wins}, cfg)
        self.assertTrue(any("벗어남" in e for e in check_plan(plan, cfg, get_layout(cfg))[0]))
        rows[0].update(start=0.0, seconds=7.5)                       # 다른 컷 구간과 겹치면 반복 오류
        rows[1].update(start=7.0, seconds=3.0)
        plan = apply_cut_edits(plan, rows, {c.id: c for c in wins}, cfg)
        self.assertTrue(any("반복" in e for e in check_plan(plan, cfg, get_layout(cfg))[0]))

    def test_quote_requires_source_and_repetition_warned(self):
        plan = self._plan(self.three(), ["笑顔が眩しすぎる", "可愛すぎる", "優しすぎる"])
        rows = caption_rows(plan)
        rows[0]["kind"] = "quote"
        plan = apply_caption_edits(plan, rows)
        errors, warns = check_plan(plan, self.cfg, self.lay)
        self.assertTrue(any("발화 근거" in e for e in errors), errors)
        self.assertTrue(any("すぎ" in w for w in warns), warns)
        self.assertTrue(repetition_warnings(["国宝級", "国宝級の美しさ"]))

    # ───────────── 수동 편집이 출력에 반영 ─────────────
    def test_p1_manual_first_clip_order_caption_in_output(self):
        clips = self.three()
        by_id = {c.id: c for c in clips}
        plan = self._plan(clips, ["一つ目", "二つ目", "三つ目"], first_clip="C")
        self.assertEqual(plan.cuts[0].clip_id, "C")
        rows = cut_rows(plan)
        rows[0]["order"], rows[1]["order"] = 2, 1
        rows[0]["start"] = 0.5
        plan = apply_cut_edits(plan, rows, by_id, self.cfg)
        crow = caption_rows(plan)
        crow[0]["text"] = "笑顔が国宝級"
        plan = apply_caption_edits(plan, crow)
        save_plan(plan, self.tmp / "edited_plan.json")
        plan = load_plan(self.tmp / "edited_plan.json")
        self.assertEqual(plan.captions[0].text, "笑顔が国宝級")
        out = self._render(plan)
        self._assert_mp4(out, plan)
        idx = frame_indices(out, 400, self.lay.video_y + 400, 280, 280)
        f0 = 0
        for s in plan.cuts:
            seg = idx[f0:f0 + s.frames]
            lo, hi = s.src_start * 30 - 0.02, (s.src_start + s.frames / 30) * 30
            self.assertTrue(all(lo <= i < hi for i in seg), (s.clip_id, seg[:3], lo, hi))
            f0 += s.frames
        job = out.parent
        used = json.loads((job / "edit_plan.json").read_text(encoding="utf-8"))
        self.assertEqual([c["text"] for c in used["captions"]], [c.text for c in plan.captions])
        info = json.loads((job / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "done")
        for f in ("script.json", "sources.json", "metadata.json", "description.txt", "config_snapshot.json"):
            self.assertTrue((job / f).exists(), f)
        self.assertNotIn("_root", json.loads((job / "config_snapshot.json").read_text(encoding="utf-8")))

    def test_failed_render_marks_job_failed(self):
        plan = self._plan(self.three(), ["一つ目"])
        cfg = copy.deepcopy(self.cfg)
        cfg["audio"]["music_path"] = str(self.tmp / "missing.wav")    # 검증 통과 후 합성 단계에서 실패
        plan.fingerprint = __import__("shorts.plan", fromlist=["x"]).fingerprint(
            cfg, pipeline.plan_sources(plan), plan.layout, plan.style)
        with self.assertRaises(PipelineError):
            pipeline.render_plan(cfg, dict(SCRIPT), plan, job_id="failjob")
        info = json.loads((Path(cfg["output_dir"]) / "テスト" / "jobs" / "failjob" / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "failed")
        self.assertIn("음원", info["error"])
        self.assertFalse(list((Path(cfg["output_dir"]) / "テスト" / "jobs" / "failjob").glob("*.mp4")))

    def test_p1_edit_out_of_bounds_and_repeat_blocked(self):
        clips = [clip("A", self.v30, 0.0, 3.0), clip("B", self.v30, 4.0, 7.9)]
        by_id = {c.id: c for c in clips}
        plan = self._plan(clips, ["一つ目", "二つ目"])
        rows = cut_rows(plan)
        rows[0]["seconds"] = 5.0
        rows[1]["clip_id"] = rows[0]["clip_id"]
        plan = apply_cut_edits(plan, rows, by_id, self.cfg)
        errors, _ = check_plan(plan, self.cfg, self.lay)
        self.assertTrue(any("벗어남" in e for e in errors), errors)
        self.assertTrue(any("반복" in e for e in errors), errors)
        with self.assertRaises(PipelineError):
            validate_for_render(plan, self.cfg, pipeline.plan_sources(plan))

    def test_p1_ui_table_nan_rows(self):
        import pandas as pd
        clips = {c.id: c for c in self.three()}
        plan = self._plan(list(clips.values()), ["x", "y"])
        df = pd.DataFrame([
            {"order": 2, "clip_id": "B", "start": 0.1, "seconds": 1.5, "role": "normal", "marker": None},
            {"order": 1.0, "clip_id": "A", "start": float("nan"), "seconds": np.float64(1.8), "role": None,
             "marker": float("nan")},
            {"order": None, "clip_id": None, "start": None, "seconds": None, "role": None, "marker": None},
        ])
        plan = apply_cut_edits(plan, df.to_dict("records"), clips, self.cfg)
        self.assertEqual([(c.clip_id, c.frames, c.role, c.marker) for c in plan.cuts],
                         [("A", 54, "normal", None), ("B", 45, "normal", None)])
        cdf = pd.DataFrame([{"id": "c01", "text": "あ", "kind": None, "first": 1, "last": float("nan"), "ko": None,
                             "scene_note": float("nan"), "source_note": None},
                            {"id": None, "text": None, "kind": None, "first": None, "last": None, "ko": None,
                             "scene_note": None, "source_note": None}])
        plan = apply_caption_edits(plan, cdf.to_dict("records"))
        self.assertEqual([(c.id, c.text, c.kind, c.first_cut, c.last_cut, c.ko, c.scene_note) for c in plan.captions],
                         [("c01", "あ", "fan", 0, 0, "", "")])

    def test_p1_stale_plan_rejected(self):
        src = self.tmp / "stale.mp4"
        shutil.copy(self.v30, src)
        plan = self._plan([clip("A", src, 0, 3.9)], ["テスト"])
        validate_for_render(plan, self.cfg, pipeline.plan_sources(plan))
        cfg = copy.deepcopy(self.cfg)
        cfg["subtitle"]["size"] = 60
        with self.assertRaisesRegex(PipelineError, "바뀌었"):
            validate_for_render(plan, cfg, pipeline.plan_sources(plan))
        plan.style = "youth_romance"                                   # 스타일만 바꿔도 오래된 계획
        with self.assertRaisesRegex(PipelineError, "바뀌었"):
            validate_for_render(plan, self.cfg, pipeline.plan_sources(plan))
        plan.style = "praise_list"
        os.utime(src, (time.time() + 5, time.time() + 5))
        with self.assertRaisesRegex(PipelineError, "바뀌었"):
            validate_for_render(plan, self.cfg, pipeline.plan_sources(plan))

    def test_p1_run_dirs_are_separate(self):
        self.assertNotEqual(pipeline.new_run_dir(self.cfg, SCRIPT), pipeline.new_run_dir(self.cfg, SCRIPT))

    def test_p1_relayout_keeps_edits(self):
        clips = self.three()[:2]
        plan = self._plan(clips, ["一つ目", "二つ目"])
        before = [(s.clip_id, s.frames) for s in plan.cuts], [c.text for c in plan.captions]
        plan = relayout(plan, "square", {c.id: c for c in clips}, self.cfg, pipeline.plan_sources(plan))
        self.assertEqual(before, ([(s.clip_id, s.frames) for s in plan.cuts], [c.text for c in plan.captions]))
        validate_for_render(plan, self.cfg, pipeline.plan_sources(plan))
        self.assertEqual(plan.cuts[0].crop[3], H)

    def test_v1_plan_converted_without_style_change(self):
        v1 = {"version": 1, "person": "p", "title_idx": 0, "title": ["a", "b"], "layout": "tall", "fps": 30,
              "slots": [{"clip_id": "A", "source": str(self.v30), "clip_start": 0.0, "clip_end": 3.9,
                         "src_start": 0.1, "frames": 45, "caption": "一つ目", "crop": [0, 0, 288, 360], "relaxed": ""}],
              "sources": [{"path": str(self.v30)}], "fingerprint": "x", "run_dir": str(self.tmp), "target": 4,
              "max_duration": 5}
        (self.tmp / "v1.json").write_text(json.dumps(v1), encoding="utf-8")
        p = load_plan(self.tmp / "v1.json")
        self.assertEqual((p.style, len(p.cuts), p.captions[0].text, p.captions[0].first_cut),
                         ("praise_list", 1, "一つ目", 0))


class TextLayoutTests(unittest.TestCase):
    def setUp(self):
        self.cfg = base_cfg(Path(tempfile.gettempdir()))

    def test_long_caption_wraps_without_loss(self):
        for preset in ("tall", "square"):
            for style in STYLES:
                cfg = styled_cfg(self.cfg, style)
                lay = get_layout(cfg, preset)
                text = "この笑顔を見るだけで今日一日がんばれる気がする件について"
                fit = fit_subtitle(text, cfg, lay)
                self.assertEqual("".join(fit.lines), text)
                self.assertGreater(len(fit.lines), 1)
                self.assertTrue(all(ln[0] not in "、。！？ーっ」』）" for ln in fit.lines), fit.lines)
                self.assertLessEqual(max(fit.widths), lay.text_w)

    def test_too_long_or_empty_caption_is_error_not_truncated(self):
        lay = get_layout(self.cfg, "tall")
        with self.assertRaises(TextFitError):
            fit_subtitle("あ" * 120, self.cfg, lay)
        with self.assertRaises(TextFitError):
            fit_subtitle("   ", self.cfg, lay)

    def test_punctuation_and_title(self):
        lay = get_layout(self.cfg, "tall")
        fit = fit_subtitle("え、待って！？この顔は反則でしょ…！", self.cfg, lay)
        self.assertEqual("".join(fit.lines), "え、待って！？この顔は反則でしょ…！")
        t = fit_title("本田翼の透明感は", "国宝級と言っても過言ではない", self.cfg, lay)
        self.assertEqual("".join(t.lines), "本田翼の透明感は国宝級と言っても過言ではない")
        long2 = ("本田翼の透明感は", "もはや国宝級と言っても過言ではない件")
        with self.assertRaises(TextFitError):
            fit_title(*long2, self.cfg, lay)
        t = fit_title(*long2, self.cfg, get_layout(self.cfg, "square"))
        self.assertEqual("".join(t.lines), "".join(long2))
        self.assertEqual(len(t.lines), 3)

    def test_rendered_pixels_stay_in_regions(self):
        from PIL import Image
        tmp = Path(tempfile.mkdtemp())
        try:
            for preset in ("tall", "square"):
                for style in STYLES:
                    cfg = styled_cfg(self.cfg, style)
                    lay = get_layout(cfg, preset)
                    sub = np.array(Image.open(render_subtitle("この笑顔を見るだけで今日一日がんばれる", cfg, lay,
                                                              tmp / f"s_{preset}.png")))[..., 3]
                    ys, xs = np.nonzero(sub)
                    self.assertGreaterEqual(ys.min(), lay.caption_y, preset)
                    self.assertLess(ys.max(), lay.height)
                    ttl = np.array(Image.open(render_title("本田翼", "この笑顔、ずるい", cfg, lay,
                                                           tmp / f"t_{preset}.png")))
                    ys, _ = np.nonzero(ttl[..., 3])
                    self.assertLess(ys.max(), lay.title_h, preset)
            # 청춘·설렘 제목: 1행 흰색, 2행 강조색
            cfg = styled_cfg(self.cfg, "youth_romance")
            img = np.array(Image.open(render_title("本田翼", "この笑顔、ずるい", cfg, get_layout(cfg, "tall"),
                                                   tmp / "accent.png")).convert("RGBA"))
            colors = {tuple(int(v) for v in c) for c in img[img[..., 3] == 255][:, :3]}
            self.assertIn((255, 255, 255), colors)
            self.assertIn((0xFF, 0xD9, 0xE4), colors)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_layout_presets_and_legacy_config(self):
        for name in ("tall", "square"):
            lay = get_layout(self.cfg, name)
            self.assertEqual(lay.title_h + lay.video_h + lay.caption_h, 1920)
        legacy = copy.deepcopy(self.cfg)
        legacy["layout"] = {"top_area": 430, "video_size": 1080, "face_fill": 0.42, "min_face_ratio": 0.04}
        lay = get_layout(legacy)
        self.assertEqual((lay.name, lay.title_h, lay.video_h, lay.caption_h), ("square", 430, 1080, 410))

    def test_config_validation_messages(self):
        bad = copy.deepcopy(self.cfg)
        bad["video"]["fps"] = 0
        bad["video"]["target_duration"] = -1
        bad["subtitle"].update(min_sec=3, max_sec=2)
        bad["layout"]["presets"]["tall"]["video_h"] = 1700
        with self.assertRaises(PipelineError) as e:
            validate_config(bad)
        msg = str(e.exception)
        for part in ("video.fps", "video.target_duration", "subtitle.min_sec", "layout.presets.tall"):
            self.assertIn(part, msg)
        validate_config(self.cfg)

    def test_crop_keeps_head_and_limits_upscale(self):
        lay = get_layout(self.cfg, "tall")
        for face in [(900, 20, 200, 240), (100, 300, 300, 360), (860, 400, 120, 140), (1700, 900, 180, 170)]:
            x, y, w, h = crop_for(face, 1920, 1080, lay, self.cfg)
            fx, fy, fw, fh = face
            self.assertAlmostEqual(w / h, lay.width / lay.video_h, delta=0.01)
            self.assertTrue(0 <= x and x + w <= 1920 and 0 <= y and y + h <= 1080, (face, (x, y, w, h)))
            self.assertLessEqual(x, fx)
            self.assertGreaterEqual(x + w, fx + fw)
            self.assertGreaterEqual(y + h, fy + fh, "턱이 잘림")
            self.assertLessEqual(y, max(fy - 0.6 * fh, 0) + 2, "머리 위 여백 부족")
            self.assertLessEqual(lay.video_h / h, self.cfg["layout"]["max_upscale"] + 0.01)


class ScriptAndSourceTests(unittest.TestCase):
    def test_old_script_loads_without_style_change(self):
        old = {"name_ja": "x", "titles": [{"line1": "a", "line2": "b", "ko": "c"}],
               "captions": [{"ja": "あ", "ko": "아"}, {"ja": None, "ko": ""}, {"ja": float("nan")}]}
        w = []
        s = director.normalize_script(old, w)
        self.assertEqual((s["theme"], s["style"], s["search_keywords_ko"]), ("", "", []))
        self.assertEqual([c["id"] for c in s["captions"]], ["c01"])
        self.assertEqual(len(w), 2)
        self.assertIn("captions[1]", w[0])
        s = director.load_script(ROOT / "examples" / "script.json")
        self.assertEqual(s["style"], "")

    def test_script_errors_name_field_and_row(self):
        with self.assertRaises(PipelineError) as e:
            director.normalize_script({"name_ja": "", "titles": [{"line1": "", "line2": None}], "captions": []})
        self.assertIn("name_ja", str(e.exception))
        self.assertIn("titles[0]", str(e.exception))

    def test_selection_saved_and_restored_by_id(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            s = director.normalize_script({"name_ja": "x", "titles": [{"line1": "a", "line2": "b"}] * 2,
                                           "captions": [{"ja": "一"}, {"ja": "二"}, {"ja": "三"}]})
            s["selection"] = {"used": ["c03", "c01"], "title_idx": 1}
            s["captions"].insert(0, {"id": "c09", "ja": "新しい", "ko": ""})   # 행 위치가 바뀌어도 ID로 복원
            director.save_script(s, tmp / "script.json")
            r = director.load_script(tmp / "script.json")                    # '재시작 후 불러오기'
            self.assertEqual(r["selection"], {"used": ["c03", "c01"], "title_idx": 1})
            self.assertEqual([c["id"] for c in r["captions"]], ["c09", "c01", "c02", "c03"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sources_json_roundtrip_keeps_attribution(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "v.mp4").write_bytes(b"x")
            src = [Source(path=str(tmp / "v.mp4"), video_id="abc", title="T", channel="Ch",
                          url="https://youtu.be/abc", license="creativeCommon")]
            sources.save_sources(src, tmp / "sources.json")
            back = sources.from_local_dir(tmp)
            self.assertEqual(back[0].attribution(), src[0].attribution())
            self.assertEqual(back[0].video_id, "abc")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
