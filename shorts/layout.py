"""화면 영역 정의와 얼굴 기준 크롭 계산.

1080x1920 캔버스를 위에서부터 [제목 | 영상 | 추가 자막] 세 영역으로 나눈다.
추가 자막은 영상 아래 전용 영역에 들어가므로 원본 영상에 박힌 자막과 겹치지 않는다.
(원본 자막은 영상 픽셀의 일부라 지워지지 않는다. 자막이 많은 소스는 편집 화면에서 제외한다.)
"""
from __future__ import annotations

from dataclasses import dataclass

from .common import PipelineError

MIN_CAPTION_H = 160      # 자막 1줄 + 상자 여백이 들어갈 최소 높이
LEGACY_PRESET = "square"


@dataclass(frozen=True)
class Layout:
    name: str
    width: int
    height: int
    title_h: int
    video_h: int
    side_margin: int

    @property
    def video_y(self) -> int:
        return self.title_h

    @property
    def caption_y(self) -> int:
        return self.title_h + self.video_h

    @property
    def caption_h(self) -> int:
        return self.height - self.caption_y

    @property
    def text_w(self) -> int:
        return self.width - 2 * self.side_margin


def presets(cfg: dict) -> dict[str, dict]:
    lay = cfg["layout"]
    if "presets" in lay:
        return lay["presets"]
    # 구버전 config: top_area + video_size(정사각)만 있음
    return {LEGACY_PRESET: {"title_h": lay.get("top_area", 430), "video_h": lay.get("video_size", 1080)}}


def get_layout(cfg: dict, name: str | None = None) -> Layout:
    v, lay = cfg["video"], cfg["layout"]
    ps = presets(cfg)
    name = name or lay.get("preset") or next(iter(ps))
    if name not in ps:
        raise PipelineError(f"레이아웃 '{name}'이(가) 없습니다. 사용 가능: {', '.join(ps)}")
    p = ps[name]
    out = Layout(name, int(v["width"]), int(v["height"]), int(p["title_h"]), int(p["video_h"]),
                 int(lay.get("side_margin", 48)))
    validate_layout(out)
    return out


def validate_layout(l: Layout) -> None:
    if l.title_h < 0 or l.video_h <= 0:
        raise PipelineError(f"레이아웃 '{l.name}': 높이 값이 잘못되었습니다.")
    if l.caption_h < MIN_CAPTION_H:
        raise PipelineError(
            f"레이아웃 '{l.name}': 자막 영역 높이 {l.caption_h}px (제목 {l.title_h} + 영상 {l.video_h} / 전체 "
            f"{l.height})이 최소 {MIN_CAPTION_H}px보다 작습니다.")
    if l.video_h % 2 or l.width % 2:
        raise PipelineError(f"레이아웃 '{l.name}': 영상 크기는 짝수여야 합니다 (H.264).")
    if not 0 <= l.side_margin < l.width // 4:
        raise PipelineError(f"레이아웃 '{l.name}': side_margin이 너무 큽니다.")


def crop_for(face: tuple[int, int, int, int] | None, fw: int, fh: int, lay: Layout, cfg: dict
             ) -> tuple[int, int, int, int]:
    """원본(fw x fh)에서 영상 영역 비율(lay.width:lay.video_h)의 크롭 (x, y, w, h).

    - 얼굴 높이 / 크롭 높이 ≈ face_fill ('얼빡' 확대)
    - 확대 배율은 max_upscale 이하 (화질 보호)
    - 얼굴 박스 위로 headroom × 얼굴 높이를 남겨 머리가 잘리지 않게, 얼굴이 크롭 안에 다 들어가게
    """
    c = cfg["layout"]
    aspect = lay.width / lay.video_h
    max_h = min(fh, fw / aspect)                         # 원본 안에 들어가는 가장 큰 크롭
    if face is None:
        h = max_h
        w = h * aspect
        return _even(int((fw - w) / 2)), _even(int((fh - h) / 2)), _even(int(w)), _even(int(h))
    x, y, w_face, h_face = face
    head = float(c.get("headroom", 0.6))
    min_h = min(max_h, lay.video_h / float(c.get("max_upscale", 2.5)))
    need_h = h_face * (1 + head + 0.35)                   # 머리카락 + 얼굴 + 턱 아래 약간
    h = min(max(h_face / float(c.get("face_fill", 0.42)), need_h, min_h), max_h)
    w = h * aspect
    if w < w_face:                                        # 아주 가까운 얼굴: 얼굴 폭이 다 들어가게
        w = min(w_face * 1.1, max_h * aspect)
        h = w / aspect
    cx, cy = x + w_face / 2, y + h_face / 2
    x0 = min(max(cx - w / 2, 0), fw - w)
    y0 = min(cy - 0.42 * h, y - head * h_face)            # 얼굴은 위쪽 42% 근처, 단 머리 위 여백 우선
    y0 = min(max(y0, 0), fh - h)
    return _even(int(x0)), _even(int(y0)), _even(int(w)), _even(int(h))


def _even(v: int) -> int:
    return max(v - v % 2, 0)


def crop_quality(crop: tuple[int, int, int, int], lay: Layout) -> float:
    """확대 배율 (1.0 = 원본 픽셀 그대로, 2.0 = 두 배 확대)."""
    return lay.video_h / max(crop[3], 1)
