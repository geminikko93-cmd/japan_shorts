"""편집 계획(edit_plan.json): 영상 컷 타임라인과 자막 타임라인을 따로 가진다.

- VideoClipEvent: 어느 장면의 어느 구간을 몇 프레임. 장면 경계 안에서만 고른다
    clip.start <= src_start, src_start + frames/fps <= clip.end  (허용 오차 BOUND_EPS)
- CaptionEvent: 'first_cut번째 컷 ~ last_cut번째 컷' 동안 표시. 컷 0개(무자막)·여러 개에 걸친 자막 모두 가능
  컷 순서를 바꿔도 자막은 위치(몇 번째 컷)에 남는다 → 컷은 그대로 두고 자막만 바꿀 수 있음
- 장면이 모자라면 반복/정지 화면으로 채우지 않고 짧게 완성하고 이유를 notes에 남김
- 반복 판별: 같은 소스 시간 구간 중첩(항상 금지) + 대표 프레임 지각 해시(비슷하면 뒤로 미룸)
  같은 인물이라는 이유(얼굴 임베딩)로는 중복 처리하지 않음
- 자막과 장면은 내용(웃음/연기 등)을 보고 맞추지 않는다 → 편집 화면에서 사람이 고침
- 버전 1(자막 1개 = 컷 1개 Slot) 계획은 불러올 때 변환한다 (스타일은 praise_list로 유지)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .common import PipelineError, log
from .layout import Layout, crop_for, crop_quality, get_layout
from .scenes import Clip, hamming
from .sources import Source
from .styles import LEGACY_STYLE, get_style, styled_cfg
from .textrender import TextFitError, fit_subtitle, fit_title, reading_seconds, subtitle_seconds

PLAN_VERSION = 2
LEAD_SEC = 0.08          # 장면 전환 직후 프레임(이전 장면 잔상)을 피하는 최소 시작 여유
BOUND_EPS = 1e-3         # 경계 비교 허용 오차(초). 시각은 1/10000초 단위로 저장됨
KINDS = {"fan": "팬 코멘트", "quote": "원본 대사"}


@dataclass
class VideoClipEvent:
    clip_id: str
    source: str            # 원본 파일 경로
    scene_start: float     # 장면 경계 (검증용)
    scene_end: float
    src_start: float       # 실제 재생 시작 (원본 프레임 격자에 맞춤)
    frames: int            # 출력 프레임 수 → 길이 = frames / fps
    crop: tuple[int, int, int, int]
    role: str = "normal"   # normal / key (핵심 표정)
    marker: float | None = None   # 사람이 지정한 표정 마커 (원본 시각). 자동 인식 아님
    relaxed: str = ""      # 중복 제한을 완화해서 고른 컷이면 이유


@dataclass
class CaptionEvent:
    id: str
    text: str
    first_cut: int         # 0부터 (포함)
    last_cut: int          # 0부터 (포함)
    kind: str = "fan"      # fan(팬 코멘트) / quote(원본 대사: source_note에 확인한 발화 근거 필요)
    ko: str = ""           # 한국어 뜻
    ko_for: str = ""       # ko가 어떤 일본어 문구 기준인지 (text와 다르면 뜻이 오래됨)
    scene_note: str = ""   # 사용자가 확인한 장면 설명 (영상 분석 결과 아님)
    source_note: str = ""  # quote일 때 발화 근거 (예: 0:12 인터뷰 발화 직접 확인)
    review: dict | None = None   # AI 검수 결과 (사람 검수 아님)


@dataclass
class EditPlan:
    person: str
    style: str
    title_idx: int
    title: tuple[str, str]
    layout: str
    fps: int
    cuts: list[VideoClipEvent]
    captions: list[CaptionEvent]
    sources: list[dict]                  # Source 메타데이터 (출처 표기용)
    fingerprint: str                     # 설정·소스·스타일이 바뀌면 달라짐 → 오래된 계획 재사용 방지
    run_dir: str                         # 실행별 작업 폴더 (썸네일·세그먼트)
    target: float
    max_duration: float
    first_clip: str = ""
    excluded_sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    audio: dict = field(default_factory=dict)   # 기준 곡·파일·오프셋·마커·타이밍 확인 상태
    title_review: dict | None = None
    source_stats: list = field(default_factory=list)   # 계획을 만들 때 소스 파일 [경로, 크기, 수정시각]
    version: int = PLAN_VERSION
    created: str = ""

    @property
    def total_frames(self) -> int:
        return sum(c.frames for c in self.cuts)

    @property
    def total(self) -> float:
        return self.total_frames / self.fps

    def cut_starts(self) -> list[int]:
        out, f = [], 0
        for c in self.cuts:
            out.append(f)
            f += c.frames
        return out + [f]

    def caption_frames(self, cap: CaptionEvent) -> tuple[int, int]:
        """(시작 프레임, 끝 프레임 - 미포함)."""
        st = self.cut_starts()
        return st[cap.first_cut], st[cap.last_cut + 1]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "EditPlan":
        d = dict(d)
        if d.get("version") == 1:
            d = upgrade_v1(d)
        if d.get("version") != PLAN_VERSION:
            raise PipelineError(f"edit_plan 버전을 알 수 없습니다 ({d.get('version')}). 다시 분석하세요.")
        d["cuts"] = [VideoClipEvent(**{**c, "crop": tuple(c["crop"])}) for c in d["cuts"]]
        d["captions"] = [CaptionEvent(**c) for c in d["captions"]]
        d["title"] = tuple(d["title"])
        return cls(**d)


def upgrade_v1(d: dict) -> dict:
    """버전 1(Slot: 자막 1개 = 컷 1개) → 버전 2. 스타일은 기존 동작(praise_list)으로 둔다."""
    cuts, caps = [], []
    for i, s in enumerate(d["slots"]):
        cuts.append({"clip_id": s["clip_id"], "source": s["source"], "scene_start": s["clip_start"],
                     "scene_end": s["clip_end"], "src_start": s["src_start"], "frames": s["frames"],
                     "crop": s["crop"], "relaxed": s.get("relaxed", "")})
        caps.append({"id": f"s{i + 1:02d}", "text": s["caption"], "first_cut": i, "last_cut": i})
    out = {k: v for k, v in d.items() if k != "slots"}
    out.update(version=PLAN_VERSION, style=LEGACY_STYLE, cuts=cuts, captions=caps, audio={}, title_review=None)
    return out


def save_plan(plan: EditPlan, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.to_json(), encoding="utf-8")
    return path


def load_plan(path: Path) -> EditPlan:
    try:
        return EditPlan.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as e:
        raise PipelineError(f"편집 계획 파일을 읽을 수 없습니다: {path} ({type(e).__name__}: {e})") from e


# ───────────────────────── 설정 검증 / 지문 ─────────────────────────
def durations(cfg: dict) -> tuple[float, float]:
    v = cfg["video"]
    target = float(v.get("target_duration", 30))
    maxd = float(v.get("max_duration", target))
    if target <= 0 or maxd <= 0:
        raise PipelineError("video.target_duration / max_duration은 0보다 커야 합니다.")
    if target > maxd:
        raise PipelineError(f"video.target_duration({target}) > max_duration({maxd}) 입니다. 설정을 고치세요.")
    return target, maxd


def source_stats(sources: list[Source]) -> list:
    out = []
    for s in sources:
        try:
            st = Path(s.path).stat()
            out.append([str(Path(s.path).resolve()), st.st_size, st.st_mtime_ns])
        except OSError:
            out.append([str(s.path), None, None])
    return sorted(out)


def is_stale(plan: "EditPlan", cfg: dict, sources: list[Source]) -> bool:
    return plan.fingerprint != fingerprint(cfg, sources, plan.layout, plan.style)


def sources_unchanged(plan: "EditPlan", sources: list[Source]) -> bool:
    return bool(plan.source_stats) and plan.source_stats == source_stats(sources)


def revalidate(plan: "EditPlan", clips: dict[str, Clip], cfg: dict, sources: list[Source]) -> "EditPlan":
    """설정만 바뀌었고 소스 파일이 그대로일 때: 편집(컷·자막)은 유지하고 크롭·지문만 현재 설정으로 다시 계산.

    소스 파일이 바뀌었으면 장면 경계를 믿을 수 없으므로 거부한다 (다시 분석해야 함).
    """
    if not sources_unchanged(plan, sources):
        raise PipelineError("소스 파일이 바뀌었거나 이전 형식 계획이라 편집을 유지할 수 없습니다. 다시 분석하세요.")
    relayout(plan, plan.layout, clips, cfg, sources)
    plan.notes.append(f"{time.strftime('%Y-%m-%d %H:%M')} 현재 설정으로 다시 검증 (편집 유지)")
    return plan


def fingerprint(cfg: dict, sources: list[Source], layout_name: str, style: str) -> str:
    """렌더 결과에 영향을 주는 설정 + 스타일 + 소스 파일 상태. 하나라도 바뀌면 다른 값."""
    scfg = styled_cfg(cfg, style)
    keys = {k: scfg.get(k) for k in ("video", "title", "subtitle", "timeline")}
    keys["layout"] = {**cfg["layout"], "preset": layout_name}
    keys["style"] = get_style(cfg, style)
    keys["scene_threshold"] = cfg.get("sources", {}).get("scene_threshold")
    au = cfg.get("audio", {})
    keys["audio"] = {k: au.get(k) for k in ("music_path", "music_start", "music_volume")}
    files = []
    for p in [s.path for s in sources] + ([au["music_path"]] if au.get("music_path") else []):
        try:
            st = Path(p).stat()
            files.append([str(Path(p).resolve()), st.st_size, st.st_mtime_ns])
        except OSError:
            files.append([str(p), None, None])
    blob = json.dumps([keys, sorted(files)], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ───────────────────────── 구간 계산 ─────────────────────────
def snap_start(t: float, src_fps: float) -> float:
    """t 이후 첫 원본 프레임 시각. 1/10000초로 '내림'해 저장 → ffmpeg 정확 탐색이 그 프레임을 첫 프레임으로 씀.

    (올림하면 저장값이 프레임 시각보다 커져 그 프레임이 버려지고 한 프레임 늦게 시작한다.)
    프레임 시각의 반올림 오차(1/100 프레임 이하)는 같은 프레임으로 본다.
    """
    k = math.ceil(t * src_fps - 0.01)
    return math.floor(k / src_fps * 1e4) / 1e4


def start_at(clip: Clip, t: float) -> float:
    """원본 시각 t를 장면 안의 재생 시작으로. 장면 첫 LEAD_SEC(전환 잔상)는 피함."""
    return snap_start(max(t, clip.scene_start + LEAD_SEC), clip.fps)


def usable(clip: Clip) -> tuple[float, float]:
    """후보 구간 시작에서 재생할 때 (시작, 장면 끝까지 쓸 수 있는 길이). 장면 경계는 넘지 않음."""
    start = start_at(clip, clip.start)
    return start, clip.scene_end - start


def make_cut(clip: Clip, start: float, frames: int, lay: Layout, cfg: dict, role: str = "normal",
             marker: float | None = None, relaxed: str = "") -> VideoClipEvent:
    """start = 원본 시각 (장면 기준이 아님)."""
    fw, fh = clip.frame_size
    return VideoClipEvent(clip.id, clip.source.path, clip.scene_start, clip.scene_end, start_at(clip, start), frames,
                          crop_for(clip.face, fw, fh, lay, cfg), role, marker, relaxed)


def propose_around_marker(clip: Clip, marker: float, cfg: dict, style: str) -> tuple[float, float]:
    """사람이 지정한 표정 마커(원본 시각) 앞뒤 맥락을 포함한 구간 → (장면 기준 시작 초, 길이 초).

    같은 장면 안에서만 잡고, 장면이 짧으면 늘리지 않는다 (슬로모션·정지 화면 없음).
    """
    if not clip.scene_start <= marker <= clip.scene_end:
        raise PipelineError(f"{clip.id}: 마커 {marker:.2f}초가 장면 {clip.scene_start:.2f}~{clip.scene_end:.2f}초 밖입니다.")
    c = get_style(cfg, style)["cut"]
    st = start_at(clip, marker - float(c.get("marker_before_sec", 1.0)))
    b = min(clip.scene_end, marker + float(c.get("marker_after_sec", 2.5)))
    length = b - st
    if length < float(c.get("min_sec", 1.5)) - 1e-6:
        raise PipelineError(f"{clip.id}: 마커 주변에 쓸 수 있는 길이가 {length:.2f}초뿐입니다 (장면이 짧음).")
    return round(st - clip.scene_start, 4), round(length, 3)


def caption_seconds(text: str, cfg: dict, lay: Layout) -> tuple[float, str]:
    """기본 표시 시간과 경고(문구 수정 필요/읽기 시간 부족)."""
    try:
        n_lines = len(fit_subtitle(text, cfg, lay).lines)
        warn = ""
    except TextFitError:
        n_lines, warn = 2, "문구가 자막 영역에 안 들어감 → 줄이기"
    sec = subtitle_seconds(text, cfg, n_lines)
    return sec, warn


def overlaps(a: tuple, b_src: str, b0: float, b1: float) -> bool:
    src, a0, a1 = a
    return src == b_src and a0 < b1 - BOUND_EPS and b0 < a1 - BOUND_EPS


# ───────────────────────── 자동 배치 ─────────────────────────
class _Picker:
    """중복을 피해 장면을 고른다. 같은 소스의 겹치는 구간은 절대 안 쓰고, 해시가 비슷하면 뒤로 미룸."""

    def __init__(self, pool: list[Clip], cfg: dict):
        tl = cfg.get("timeline", {})
        self.pool = pool
        self.sim_d = int(tl.get("similar_hash_distance", 10))
        self.allow_similar = bool(tl.get("allow_similar_fallback", True))
        self.used_ids: set[str] = set()
        self.used_iv: list[tuple[str, float, float]] = []
        self.used_hash: list[int] = []

    def nearest(self, c: Clip) -> int:
        return min((hamming(c.dhash, h) for h in self.used_hash), default=64)

    def candidates(self, need: float) -> list[Clip]:
        out = []
        for c in self.pool:
            st, room = usable(c)
            if c.id not in self.used_ids and room + BOUND_EPS >= need and \
                    not any(overlaps(u, c.source.path, st, st + need) for u in self.used_iv):
                out.append(c)
        return out

    def pick(self, need: float, prefer=None) -> tuple[Clip | None, str]:
        cands = self.candidates(need)
        if prefer:
            cands = sorted(cands, key=prefer)
        fresh = [c for c in cands if self.nearest(c) > self.sim_d]
        if fresh:
            return fresh[0], ""
        if cands and self.allow_similar:
            c = max(cands, key=self.nearest)
            return c, f"비슷한 장면 허용 (다른 장면 부족, 해시 거리 {self.nearest(c)})"
        return None, ""

    def use(self, c: Clip, start: float, dur: float) -> None:
        self.used_ids.add(c.id)
        self.used_iv.append((c.source.path, start, start + dur))
        self.used_hash.append(c.dhash)


def _norm_captions(captions: list) -> list[dict]:
    out = []
    for i, c in enumerate(captions):
        if isinstance(c, str):
            c = {"id": f"x{i + 1:02d}", "ja": c}
        text = str(c.get("ja") or c.get("text") or "").strip()
        if text:
            out.append({"id": c.get("id") or f"x{i + 1:02d}", "text": text, "ko": c.get("ko") or "",
                        "kind": c.get("kind") or "fan"})
    return out


def auto_plan(clips: list[Clip], captions: list, cfg: dict, lay: Layout, *, style: str = LEGACY_STYLE,
              first_clip: str = "", climax_clip: str = "", exclude_sources: list[str] | None = None
              ) -> tuple[list[VideoClipEvent], list[CaptionEvent], list[str]]:
    """스타일 프리셋에 따라 컷과 자막을 배치. (컷, 자막, 메모) 반환. 내용 기반 매칭은 하지 않는다."""
    fps = int(cfg["video"]["fps"])
    target, maxd = durations(cfg)
    st = get_style(cfg, style)
    scfg = styled_cfg(cfg, style)
    caps = _norm_captions(captions)
    if not caps:
        raise PipelineError("사용할 자막이 없습니다. ② 탭에서 칭찬글을 체크하세요.")
    excl = set(exclude_sources or [])
    pool = [c for c in clips if c.source.path not in excl]
    if not pool:
        raise PipelineError("사용할 수 있는 장면이 없습니다 (모든 소스가 제외되었거나 얼굴 장면 없음).")
    by_id = {c.id: c for c in pool}
    for name, cid in (("첫 컷", first_clip), ("핵심 컷", climax_clip)):
        if cid and cid not in by_id:
            raise PipelineError(f"{name} '{cid}'을(를) 찾을 수 없습니다 (제외된 소스이거나 다른 분석 결과).")
    picker = _Picker(pool, cfg)
    min_cut = float(st["cut"].get("min_sec", 1.5))
    notes: list[str] = []

    if st["captions"]["mode"] == "per_cut":
        cuts, events, stop = _plan_per_cut(caps, picker, by_id, first_clip, scfg, lay, fps, target, maxd)
    else:
        cuts, stop = _plan_cuts(st, picker, by_id, first_clip, climax_clip, scfg, lay, fps, target, maxd, min_cut)
        events = _hold_captions(caps, cuts, st, scfg, lay, fps, notes) if cuts else []
    if not cuts:
        raise PipelineError(f"컷을 하나도 만들 수 없습니다: {stop or '장면이 너무 짧음'}")
    total = sum(c.frames for c in cuts) / fps
    if total < target - 1e-6:
        notes.insert(0, f"목표 {target:.0f}초보다 짧게 완성: {total:.1f}초 — {stop}")
    n_rel = sum(1 for c in cuts if c.relaxed)
    if n_rel:
        notes.append(f"중복 제한 완화: {n_rel}개 컷이 이미 쓴 장면과 비슷함 (편집 화면에서 교체 가능)")
    log.info("타임라인(%s): 컷 %d개, 자막 %d개, 총 %.2f초 %s", style, len(cuts), len(events), total, " / ".join(notes))
    return cuts, events, notes


def _plan_per_cut(caps, picker, by_id, first_clip, cfg, lay, fps, target, maxd):
    """기존 방식: 자막 1개 = 컷 1개, 컷 길이 = 자막 읽기 시간."""
    cuts, events, total, stop = [], [], 0, ""
    for i, c in enumerate(caps):
        if total / fps >= target - 1e-6:
            break
        sec, _ = caption_seconds(c["text"], cfg, lay)
        n = max(round(sec * fps), 1)
        if (total + n) / fps > maxd + 1e-9:
            n = int(math.floor(maxd * fps - total + 1e-6))
            if n / fps < cfg["subtitle"]["min_sec"]:
                stop = f"최대 길이 {maxd:.0f}초 도달"
                break
        dur = n / fps
        if i == 0 and first_clip:
            clip, why = by_id[first_clip], ""
            if clip not in picker.candidates(dur):
                raise PipelineError(f"첫 컷 '{first_clip}' 장면이 첫 자막 길이({dur:.2f}초)보다 짧습니다.")
        else:
            clip, why = picker.pick(dur)
        if clip is None:
            stop = "사용 가능한 장면 부족: 남은 장면이 이미 쓴 장면과 비슷하거나 자막 시간보다 짧음"
            break
        cut = make_cut(clip, clip.start, n, lay, cfg, "normal", None, why)
        picker.use(clip, cut.src_start, dur)
        cuts.append(cut)
        events.append(CaptionEvent(c["id"], c["text"], len(cuts) - 1, len(cuts) - 1, c["kind"], c["ko"],
                                   c["text"] if c["ko"] else ""))
        total += n
    else:
        if total / fps < target - 1e-6:
            stop = f"선택한 자막 {len(caps)}개를 모두 사용"
    return cuts, events, stop


def _plan_cuts(st, picker, by_id, first_clip, climax_clip, cfg, lay, fps, target, maxd, min_cut):
    """스타일 템플릿에 따라 컷 길이를 정하고 장면을 고름 (자막과 독립)."""
    cut_cfg = st["cut"]
    template = st.get("template") or [{"until": 999, "cut": "normal"}]
    cuts: list[VideoClipEvent] = []
    total, stop = 0, ""
    while total / fps < target - 1e-6:
        t = total / fps
        sec = next((s for s in template if t < s["until"] - 1e-6), template[-1])
        role = sec.get("cut", "normal")
        want = float(cut_cfg.get("key_sec" if role == "key" else "normal_sec", 2.5))
        want = max(min(want, target - t), min_cut)        # 마지막 컷은 목표 길이에 맞춤 (최소 길이 이상)
        want = min(want, maxd - t)
        if want < min_cut - 1e-6:
            stop = f"최대 길이 {maxd:.0f}초 도달"
            break
        clip, why = None, ""
        if not cuts and first_clip:
            clip = by_id[first_clip]
            if clip not in picker.candidates(min_cut):
                raise PipelineError(f"첫 컷 '{first_clip}' 장면이 너무 짧습니다 (최소 {min_cut}초).")
        elif role == "key" and cuts and climax_clip and climax_clip not in picker.used_ids:
            clip = by_id[climax_clip]
            if clip not in picker.candidates(min_cut):
                clip = None
        if clip is None:
            # 원하는 길이를 다 쓸 수 있는 장면 우선, 없으면 최소 길이 이상인 장면 (핵심 컷은 긴 장면 우선)
            prefer = (lambda c: -usable(c)[1]) if role == "key" and cuts else None
            clip, why = picker.pick(want, prefer)
            if clip is None:
                clip, why = picker.pick(min_cut, prefer)
        if clip is None:
            stop = "사용 가능한 장면 부족 (남은 장면이 이미 쓴 장면과 겹치거나 너무 짧음)"
            break
        s0, room = usable(clip)
        n = int(math.floor(min(want, room) * fps + 1e-6))
        if total + n > math.floor(maxd * fps + 1e-6):
            n = int(math.floor(maxd * fps + 1e-6)) - total
        cut = make_cut(clip, clip.start, n, lay, cfg, role, None, why)
        picker.use(clip, cut.src_start, n / fps)
        cuts.append(cut)
        total += n
    return cuts, stop


def _hold_captions(caps, cuts, st, cfg, lay, fps, notes) -> list[CaptionEvent]:
    """코멘트 K개를 컷 경계에 맞춰 고르게 나눠 여러 컷에 걸쳐 유지. 컷보다 자막이 많으면 앞에서부터만 사용."""
    leave_key = bool(st["captions"].get("leave_key_cuts_empty"))
    slots = [i for i, c in enumerate(cuts) if not (leave_key and c.role == "key" and i > 0)]
    if not slots:
        return []
    k = min(len(caps), len(slots))
    if len(caps) > k:
        notes.append(f"자막 {len(caps)}개 중 {k}개만 배치 (컷 {len(slots)}개뿐). 나머지는 편집 화면에서 추가 가능")
    # 자막 가능한 컷들을 프레임 기준으로 k등분 (각 묶음 최소 1컷, 컷 경계에 맞춤)
    frames = [cuts[i].frames for i in slots]
    total = sum(frames)
    cum = [0]
    for f in frames:
        cum.append(cum[-1] + f)
    events, j = [], 0
    for n in range(k):
        if n == k - 1:
            end = len(slots)
        else:
            goal = total * (n + 1) / k
            end = j + 1
            while end < len(slots) - (k - n - 1) and cum[end] + frames[end] / 2 <= goal:
                end += 1
        group = slots[j:end]
        j = end
        run = [group[0]]                       # 무자막 핵심 컷을 건너뛰지 않도록 연속된 컷까지만
        for g in group[1:]:
            if g != run[-1] + 1:
                break
            run.append(g)
        c = caps[n]
        events.append(CaptionEvent(c["id"], c["text"], run[0], run[-1], c["kind"], c["ko"],
                                   c["text"] if c["ko"] else ""))
    return events


# ───────────────────────── 검사 ─────────────────────────
REPEAT_PATTERNS = [("すぎ", "〜すぎる/すぎ"), ("国宝級", "国宝級"), ("尊い", "尊い"), ("神", "神"), ("最高", "最高"),
                   ("好き", "好き"), ("優勝", "優勝"), ("沼", "沼"), ("天使", "天使")]


def repetition_warnings(texts: list[str]) -> list[str]:
    """한 영상 안에서 같은 칭찬 표현·같은 끝맺음이 반복되면 경고 (다른 후보 제안용)."""
    warns = []
    for pat, label in REPEAT_PATTERNS:
        n = sum(1 for t in texts if pat in t)
        if n >= 2:
            warns.append(f"'{label}' 표현이 {n}번 반복됩니다 → 다른 후보로 바꾸는 것을 권장")
    ends = {}
    for t in texts:
        e = re.sub(r"[！!？?…。、\s]+$", "", t)[-2:]
        if e:
            ends[e] = ends.get(e, 0) + 1
    for e, n in ends.items():
        if n >= 3:
            warns.append(f"끝맺음 '…{e}'이(가) {n}번 반복됩니다")
    return warns


def check_plan(plan: EditPlan, cfg: dict, lay: Layout) -> tuple[list[str], list[str]]:
    """(렌더를 막는 오류, 경고). 오류/경고는 '컷 #n', '자막 <id>'로 시작해 표에 붙일 수 있다."""
    errors, warns = [], []
    fps = plan.fps
    scfg = styled_cfg(cfg, plan.style)
    _, maxd = durations(cfg)
    if not plan.cuts:
        errors.append("컷이 없습니다.")
    seen: list[tuple[str, float, float]] = []
    for k, c in enumerate(plan.cuts, 1):
        end = c.src_start + c.frames / fps
        if c.frames < 1:
            errors.append(f"컷 #{k} 길이가 0입니다.")
        if c.src_start < c.scene_start - BOUND_EPS or end > c.scene_end + BOUND_EPS:
            errors.append(f"컷 #{k} {c.clip_id}: 재생 구간 {c.src_start:.3f}~{end:.3f}초가 장면 "
                          f"{c.scene_start:.3f}~{c.scene_end:.3f}초를 벗어남 → 시작/길이를 줄이세요.")
        if any(overlaps(u, c.source, c.src_start, end) for u in seen):
            errors.append(f"컷 #{k} {c.clip_id}: 앞 컷과 같은 구간을 반복합니다 → 다른 장면·구간을 고르세요.")
        seen.append((c.source, c.src_start, end))
        if c.relaxed:
            warns.append(f"컷 #{k} {c.clip_id}: {c.relaxed}")
        up = crop_quality(c.crop, lay)
        if up > float(cfg["layout"].get("max_upscale", 2.5)) + 0.05:
            warns.append(f"컷 #{k} {c.clip_id}: 원본이 작아 {up:.1f}배 확대됨 (화질 저하)")
    n = len(plan.cuts)
    ids = [c.id for c in plan.captions]
    for d in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"자막 {d}: ID가 중복됩니다.")
    spans = []
    for c in plan.captions:
        tag = f"자막 {c.id}"
        if not c.text.strip():
            errors.append(f"{tag}: 문구가 비어 있습니다.")
            continue
        if not (0 <= c.first_cut <= c.last_cut < n):
            errors.append(f"{tag}: 표시 범위 컷 {c.first_cut + 1}~{c.last_cut + 1}이(가) 잘못되었습니다 (컷 {n}개).")
            continue
        f0, f1 = plan.caption_frames(c)
        spans.append((f0, f1, c.id))
        try:
            n_lines = len(fit_subtitle(c.text, scfg, lay).lines)
            need = reading_seconds(c.text, n_lines, scfg)
            if need > (f1 - f0) / fps + 1e-6:
                warns.append(f"{tag}: 읽기 시간 부족 (필요 {need:.1f}초 > 표시 {(f1 - f0) / fps:.2f}초)")
        except TextFitError as e:
            errors.append(f"{tag}: {e}")
        if c.kind == "quote" and not c.source_note.strip():
            errors.append(f"{tag}: '원본 대사'는 확인한 발화 근거(발화 근거 칸)가 필요합니다. 없으면 팬 코멘트로 바꾸세요.")
        if c.kind == "fan" and re.match(r"^[「『\"“].*[」』\"”]$", c.text.strip()):
            warns.append(f"{tag}: 팬 코멘트에 인용부호가 있어 본인 발언처럼 보일 수 있습니다.")
        if c.ko and c.ko_for and c.ko_for != c.text:
            warns.append(f"{tag}: 한국어 뜻이 예전 문구 기준입니다 (뜻 갱신 필요).")
        rv = c.review or {}
        if rv.get("status") == "ai_checked" and rv.get("text") != c.text:
            warns.append(f"{tag}: AI 검수 후 문구가 바뀌었습니다 (재검수 필요).")
    spans.sort()
    for (a0, a1, ai), (b0, b1, bi) in zip(spans, spans[1:]):
        if b0 < a1:
            errors.append(f"자막 {bi}: 자막 {ai}와 표시 구간이 겹칩니다.")
    warns += repetition_warnings([c.text for c in plan.captions])
    if plan.total > maxd + 1e-6:
        errors.append(f"총 길이 {plan.total:.2f}초가 최대 {maxd:.0f}초를 넘습니다.")
    try:
        fit_title(*plan.title, scfg, lay)
    except TextFitError as e:
        errors.append(f"제목: {e}")
    return errors, warns


def validate_for_render(plan: EditPlan, cfg: dict, sources: list[Source]) -> Layout:
    """오래된 계획/잘못된 구간이면 PipelineError. 통과하면 Layout 반환."""
    lay = get_layout(cfg, plan.layout)
    if plan.fps != int(cfg["video"]["fps"]):
        raise PipelineError("설정의 fps가 계획과 다릅니다. 다시 배치하세요.")
    if plan.fingerprint != fingerprint(cfg, sources, plan.layout, plan.style):
        raise PipelineError("설정·스타일 또는 소스 파일이 계획을 만든 뒤 바뀌었습니다. 다시 분석·배치하세요 "
                            "(오래된 계획은 사용하지 않습니다).")
    missing = [c.source for c in plan.cuts if not Path(c.source).exists()]
    if missing:
        raise PipelineError(f"소스 파일이 없습니다: {missing[0]}")
    errors, _ = check_plan(plan, cfg, lay)
    if errors:
        raise PipelineError("렌더 전에 고쳐야 할 항목:\n- " + "\n- ".join(errors))
    return lay


# ───────────────────────── 수동 편집 ─────────────────────────
def _val(v, default):
    try:
        return default if v is None or v != v else v      # NaN != NaN
    except (TypeError, ValueError):
        return default


def _text(v) -> str:
    return str(_val(v, "") or "").strip()


def cut_rows(plan: EditPlan) -> list[dict]:
    """컷 표용 행. start/marker는 장면 시작 기준 초."""
    return [{"order": k, "clip_id": c.clip_id, "start": round(c.src_start - c.scene_start, 4),
             "seconds": round(c.frames / plan.fps, 3), "role": c.role,
             "marker": None if c.marker is None else round(c.marker - c.scene_start, 3)}
            for k, c in enumerate(plan.cuts, 1)]


def caption_rows(plan: EditPlan) -> list[dict]:
    return [{"id": c.id, "text": c.text, "kind": c.kind, "first": c.first_cut + 1, "last": c.last_cut + 1,
             "ko": c.ko, "scene_note": c.scene_note, "source_note": c.source_note} for c in plan.captions]


def apply_cut_edits(plan: EditPlan, rows: list[dict], clips: dict[str, Clip], cfg: dict) -> EditPlan:
    """컷 표(순서·장면·시작·길이·역할·마커)를 반영. 자막은 위치(몇 번째 컷)를 그대로 유지한다.

    경계를 벗어나는 값은 고치지 않고 그대로 두어 check_plan이 오류로 알려 준다. 자동 배치로 덮어쓰지 않는다.
    """
    lay = get_layout(cfg, plan.layout)
    old = {c.clip_id: c for c in plan.cuts}
    cuts = []
    for r in sorted(rows, key=lambda r: float(_val(r.get("order"), 0))):
        cid = _text(r.get("clip_id"))
        if not cid:
            continue
        clip = clips.get(cid)
        if clip is None:
            raise PipelineError(f"장면 '{cid}'을(를) 찾을 수 없습니다.")
        n = max(int(round(float(_val(r.get("seconds"), 0)) * plan.fps)), 0)
        m = _val(r.get("marker"), None)
        marker = None if m is None or str(m).strip() == "" else clip.scene_start + float(m)
        prev = old.get(cid)
        start = clip.scene_start + float(_val(r.get("start"), clip.start - clip.scene_start))
        cuts.append(make_cut(clip, start, n, lay, cfg, _text(r.get("role")) or "normal", marker,
                             prev.relaxed if prev else ""))
    plan.cuts = cuts
    plan.first_clip = cuts[0].clip_id if cuts else ""
    return plan


def apply_marker_proposals(plan: EditPlan, clips: dict[str, Clip], cfg: dict) -> list[str]:
    """마커가 있는 컷의 시작·길이를 마커 앞뒤 맥락으로 다시 제안. 반환: 처리 메모."""
    msgs = []
    lay = get_layout(cfg, plan.layout)
    for k, c in enumerate(plan.cuts):
        if c.marker is None:
            continue
        clip = clips[c.clip_id]
        start_rel, length = propose_around_marker(clip, c.marker, cfg, plan.style)
        n = int(math.floor(length * plan.fps + 1e-6))
        plan.cuts[k] = make_cut(clip, clip.scene_start + start_rel, n, lay, cfg, "key", c.marker, c.relaxed)
        msgs.append(f"컷 #{k + 1} {c.clip_id}: 마커 {c.marker - clip.scene_start:.2f}초 기준 장면 {start_rel:.2f}초부터 "
                    f"{n / plan.fps:.2f}초")
    return msgs


def apply_caption_edits(plan: EditPlan, rows: list[dict]) -> EditPlan:
    """자막 표(문구·종류·표시 컷 범위·뜻·장면 설명·발화 근거)를 반영. 컷은 건드리지 않는다.

    ID는 유지하고 새 행에는 새 ID. 문구가 바뀌면 한국어 뜻은 남기되 '예전 문구 기준'으로 표시된다.
    """
    old = {c.id: c for c in plan.captions}
    used = set(old)
    caps = []
    for r in rows:
        text = _text(r.get("text"))
        cid = _text(r.get("id"))
        if not text and not cid:
            continue
        if not cid or cid in {c.id for c in caps}:
            n = 1
            while f"u{n:02d}" in used:
                n += 1
            cid = f"u{n:02d}"
            used.add(cid)
        prev = old.get(cid)
        ko = _text(r.get("ko"))
        ko_for = (prev.ko_for if prev and ko == prev.ko else text) if ko else ""
        kind = _text(r.get("kind")) or "fan"
        if kind not in KINDS:
            kind = next((k for k, v in KINDS.items() if v == kind), "fan")
        caps.append(CaptionEvent(cid, text, int(float(_val(r.get("first"), 1))) - 1,
                                 int(float(_val(r.get("last"), _val(r.get("first"), 1)))) - 1, kind, ko, ko_for,
                                 _text(r.get("scene_note")), _text(r.get("source_note")),
                                 prev.review if prev else None))
    plan.captions = caps
    return plan


def relayout(plan: EditPlan, layout_name: str, clips: dict[str, Clip], cfg: dict, sources: list[Source]) -> EditPlan:
    """레이아웃만 바꾸고 크롭을 다시 계산 (장면·자막·순서는 유지)."""
    lay = get_layout(cfg, layout_name)
    for c in plan.cuts:
        clip = clips[c.clip_id]
        c.crop = crop_for(clip.face, *clip.frame_size, lay, cfg)
    plan.layout = layout_name
    plan.fingerprint = fingerprint(cfg, sources, layout_name, plan.style)
    return plan


def default_audio(cfg: dict) -> dict:
    """음악 정보. 파일이나 사람이 확인한 마커가 없으면 타이밍은 '미확인'."""
    au = cfg.get("audio", {})
    f = au.get("music_path") or ""
    return {"file": f, "offset": float(au.get("music_start", 0)) if f else None,
            "reference": {"title": "", "artist": "", "version": ""}, "markers": [], "timing": "unverified"}


def audio_status(audio: dict) -> str:
    if not audio or audio.get("timing") != "user_marked":
        return "음악 타이밍 미확인 (음원 파일·사람이 확인한 마커 없음 — 박자에 맞췄다고 보지 마세요)"
    return f"사용자가 표시한 마커 {len(audio.get('markers', []))}개 기준 (자동 박자 분석 아님)"


def new_plan(person: str, title_idx: int, title: tuple[str, str], lay: Layout, cfg: dict, style: str,
             cuts: list[VideoClipEvent], captions: list[CaptionEvent], notes: list[str], sources: list[Source],
             run_dir: Path, first_clip: str = "", excluded: list[str] | None = None) -> EditPlan:
    target, maxd = durations(cfg)
    return EditPlan(person, style, title_idx, title, lay.name, int(cfg["video"]["fps"]), cuts, captions,
                    [asdict(s) for s in sources], fingerprint(cfg, sources, lay.name, style), str(run_dir),
                    target, maxd, first_clip, list(excluded or []), notes, default_audio(cfg),
                    source_stats=source_stats(sources), created=time.strftime("%Y-%m-%d %H:%M:%S"))


def clone(plan: EditPlan) -> EditPlan:
    return copy.deepcopy(plan)
