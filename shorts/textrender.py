"""④ 일본어 제목/자막 렌더링 (투명 PNG, 1080x1920 전체 캔버스).

- 폰트: config의 font, 비우면 OS의 일본어 고딕 볼드
- 줄바꿈: 픽셀 폭 기준. 조사(の/は 등)·BudouX 문절 경계에서만 나누고, 줄 길이가 고르게 되도록 분할
- 안 들어가면 min_size_ratio까지만 글자를 줄이고, 그래도 안 되면 TextFitError (문구를 자르지 않음)
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import PipelineError, log
from .layout import Layout

# 단정한 일본어 고딕 볼드 후보 (먼저 찾은 것 사용)
FONT_CANDIDATES = [
    "assets/fonts/NotoSansJP-Black.ttf",
    "assets/fonts/NotoSansJP-Bold.ttf",
    "C:/Windows/Fonts/YuGothB.ttc",
    "C:/Windows/Fonts/meiryob.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W8.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]
BREAK_AFTER = set("のはがをにでもと、。！!？?」』）)")
NO_LINE_START = set("、。，．・：；！？!?」』）)〕】ーぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ々〜～…")
NO_LINE_END = set("「『（(〔【")
LINE_SPACING = 1.22
BOX_PAD = 24             # 자막 배경 상자 안쪽 여백
REGION_PAD = 16          # 영역 위/아래 최소 여백


class TextFitError(PipelineError):
    """문구가 영역에 들어가지 않음 → 사용자가 문구를 줄여야 함."""


@dataclass
class Fit:
    lines: list[str]
    size: int
    stroke: int
    line_h: int
    widths: list[int]
    para: list[int]          # 각 줄이 몇 번째 입력 문단(제목 line1=0, line2=1)에서 왔는지

    @property
    def height(self) -> int:
        return self.line_h * len(self.lines)


def find_font(cfg: dict, configured: str) -> str:
    root = Path(cfg["_root"])
    for cand in ([configured] if configured else []) + FONT_CANDIDATES:
        p = Path(cand) if Path(cand).is_absolute() else root / cand
        if p.exists():
            return str(p)
    raise PipelineError(
        f"일본어 폰트를 찾지 못했습니다 ({platform.system()}). Noto Sans JP를 받아 "
        "assets/fonts/NotoSansJP-Black.ttf 로 넣거나 config.yaml의 font를 지정하세요.")


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=1)
def _budoux():
    try:
        import budoux
        return budoux.load_default_japanese_parser()
    except ImportError:
        return None


def break_points(text: str) -> list[int]:
    """줄을 나눌 수 있는 위치(문자 인덱스). 금칙(줄 머리에 。、ー 등 금지) 적용."""
    pts = set()
    parser = _budoux()
    if parser:
        pos = 0
        for chunk in parser.parse(text)[:-1]:
            pos += len(chunk)
            pts.add(pos)
    pts |= {i + 1 for i, ch in enumerate(text[:-1]) if ch in BREAK_AFTER}
    if text.isascii():
        pts |= {i + 1 for i, ch in enumerate(text[:-1]) if ch == " "}
    ok = [p for p in sorted(pts) if 0 < p < len(text)
          and text[p] not in NO_LINE_START and text[p - 1] not in NO_LINE_END]
    return ok


def _width(text: str, font, stroke: int) -> int:
    if not text:
        return 0
    b = font.getbbox(text, stroke_width=stroke)
    return b[2] - b[0]


def _wrap(text: str, font, stroke: int, max_w: int, max_lines: int) -> list[str] | None:
    """가장 긴 줄이 가장 짧아지도록 문절 경계에서 n줄로 분할 (n = 1..max_lines 중 들어가는 최소)."""
    if _width(text, font, stroke) <= max_w:
        return [text]
    pts = break_points(text)
    if not pts:                       # 문절 경계가 없으면 글자 단위(금칙만 지킴)
        pts = [i for i in range(1, len(text)) if text[i] not in NO_LINE_START]
    cuts = [0, *pts, len(text)]
    n_cut = len(cuts)
    for n in range(2, max_lines + 1):
        # best[k][j] = cuts[0]..cuts[j]를 k줄로 나눴을 때 최대 줄 폭의 최소값
        INF = float("inf")
        best = [[INF] * n_cut for _ in range(n + 1)]
        prev = [[-1] * n_cut for _ in range(n + 1)]
        best[0][0] = 0
        for k in range(1, n + 1):
            for j in range(1, n_cut):
                for i in range(j):
                    if best[k - 1][i] == INF:
                        continue
                    w = _width(text[cuts[i]:cuts[j]], font, stroke)
                    m = max(best[k - 1][i], w)
                    if m < best[k][j]:
                        best[k][j], prev[k][j] = m, i
        if best[n][n_cut - 1] <= max_w:
            lines, j = [], n_cut - 1
            for k in range(n, 0, -1):
                i = prev[k][j]
                lines.append(text[cuts[i]:cuts[j]])
                j = i
            return lines[::-1]
    return None


def fit_text(paragraphs: list[str], font_path: str, size: int, stroke: int, max_w: int, max_h: int,
             min_ratio: float = 0.75) -> Fit:
    """각 문단(제목은 line1/line2)을 폭 max_w, 높이 max_h 안에 맞춤. 실패 시 TextFitError."""
    paragraphs = [p.strip().replace("\n", "") for p in paragraphs if p and p.strip()]
    if not paragraphs:
        raise TextFitError("빈 문구입니다.")
    min_size = max(int(size * min_ratio), 12)
    s = size
    while s >= min_size:
        st = max(round(stroke * s / size), 1) if stroke else 0
        font = _font(font_path, s)
        line_h = int(s * LINE_SPACING) + 2 * st
        max_lines = max(max_h // line_h, 1)
        lines: list[str] = []
        para: list[int] = []
        for pi, p in enumerate(paragraphs):
            got = _wrap(p, font, st, max_w, max_lines - len(lines))
            if got is None:
                lines = []
                break
            lines += got
            para += [pi] * len(got)
        if lines and len(lines) * line_h <= max_h:
            if "".join(lines) != "".join(paragraphs):        # 줄바꿈이 문구를 훼손하지 않았는지
                raise TextFitError(f"줄바꿈 중 문구가 바뀌었습니다: {paragraphs} → {lines}")
            return Fit(lines, s, st, line_h, [_width(ln, font, st) for ln in lines], para)
        s -= 2
    raise TextFitError(f"문구가 너무 깁니다 (글자 {min_size}px까지 줄여도 안 들어감): {' / '.join(paragraphs)}")


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _draw(img: Image.Image, fit: Fit, font_path: str, fill: str | list[str], stroke_color: str,
          y0: int, y1: int, box: list[int] | None) -> tuple[int, int, int, int]:
    """영역 [y0, y1)의 세로 가운데에 그림. 그린 상자(x0, y0, x1, y1)를 돌려줌."""
    W = img.width
    font = _font(font_path, fit.size)
    pad = BOX_PAD if box else 0
    top = (y0 + y1 - fit.height) // 2
    bw = max(fit.widths) + 2 * pad
    rect = ((W - bw) // 2, top - pad // 2, (W + bw) // 2, top + fit.height + pad // 2)
    if box:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rounded_rectangle(rect, radius=18, fill=tuple(box))
        img.alpha_composite(overlay)
    draw = ImageDraw.Draw(img)
    fills = fill if isinstance(fill, list) else [fill]
    for i, (ln, w) in enumerate(zip(fit.lines, fit.widths)):
        bb = font.getbbox(ln, stroke_width=fit.stroke)
        x = (W - w) // 2 - bb[0]
        y = top + i * fit.line_h + (fit.line_h - (bb[3] - bb[1])) // 2 - bb[1]
        color = fills[min(fit.para[i], len(fills) - 1)]
        draw.text((x, y), ln, font=font, fill=_hex(color), stroke_width=fit.stroke, stroke_fill=_hex(stroke_color))
    return rect


def fit_title(line1: str, line2: str, cfg: dict, lay: Layout) -> Fit:
    t = cfg["title"]
    return fit_text([line1, line2], find_font(cfg, t["font"]), t["size"], t["stroke_width"],
                    lay.text_w, lay.title_h - 2 * REGION_PAD, t.get("min_size_ratio", 0.75))


def fit_subtitle(text: str, cfg: dict, lay: Layout) -> Fit:
    s = cfg["subtitle"]
    pad = BOX_PAD if s.get("box_color") else 0
    return fit_text([text], find_font(cfg, s["font"]), s["size"], s["stroke_width"],
                    lay.text_w - 2 * pad, lay.caption_h - 2 * REGION_PAD - pad,
                    s.get("min_size_ratio", 0.8))


def render_title(line1: str, line2: str, cfg: dict, lay: Layout, out: Path) -> Path:
    t = cfg["title"]
    fit = fit_title(line1, line2, cfg, lay)
    img = Image.new("RGBA", (lay.width, lay.height), (0, 0, 0, 0))
    colors = [t["color"], t.get("accent_color") or t["color"]]   # 이름(1행)은 기본색, 감정 포인트(2행)는 강조색
    rect = _draw(img, fit, find_font(cfg, t["font"]), colors, t["stroke_color"], 0, lay.title_h, None)
    _check_inside(rect, (0, 0, lay.width, lay.title_h), "제목")
    img.save(out)
    return out


def render_subtitle(text: str, cfg: dict, lay: Layout, out: Path) -> Path:
    s = cfg["subtitle"]
    fit = fit_subtitle(text, cfg, lay)
    img = Image.new("RGBA", (lay.width, lay.height), (0, 0, 0, 0))
    rect = _draw(img, fit, find_font(cfg, s["font"]), s["color"], s["stroke_color"],
                 lay.caption_y, lay.height, s.get("box_color"))
    _check_inside(rect, (0, lay.caption_y, lay.width, lay.height), "자막")
    img.save(out)
    log.debug("자막 렌더: %s → %s (%dpx)", text, fit.lines, fit.size)
    return out


def _check_inside(rect, area, what: str) -> None:
    x0, y0, x1, y1 = rect
    ax0, ay0, ax1, ay1 = area
    if x0 < ax0 or y0 < ay0 or x1 > ax1 or y1 > ay1:
        raise TextFitError(f"{what} 상자가 영역을 벗어납니다: {rect} ⊄ {area}")


def reading_seconds(text: str, n_lines: int, cfg: dict) -> float:
    """읽는 데 필요한 시간 = 0.5초 + 글자수/속도 + 줄 추가당 0.3초."""
    s = cfg["subtitle"]
    cps = float(s.get("chars_per_sec", 8.0))
    return 0.5 + len(text.strip()) / cps + 0.3 * max(n_lines - 1, 0)


def subtitle_seconds(text: str, cfg: dict, n_lines: int = 1) -> float:
    """기본 표시 시간: 읽기 시간을 min_sec~max_sec로 제한. max_sec보다 더 필요하면 편집 화면에서 경고."""
    s = cfg["subtitle"]
    return round(min(max(reading_seconds(text, n_lines, cfg), s["min_sec"]), s["max_sec"]), 2)
