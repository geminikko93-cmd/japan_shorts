"""공통 유틸: 설정 로드, 로깅, ffmpeg 실행, 예외 타입."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import yaml

log = logging.getLogger("shorts")


class PipelineError(RuntimeError):
    """사용자에게 그대로 보여줄 수 있는 파이프라인 오류."""


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dotenv(path: str | Path) -> None:
    """KEY=VALUE 형식의 .env를 환경변수로 로드. 같은 이름이 이미 있어도 .env 값이 우선."""
    path = Path(path)
    if not path.is_file():
        return
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key:
            log.warning(".env %d번째 줄 무시 (KEY=VALUE 형식 아님)", n)
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif " #" in val:               # 따옴표 없는 값 뒤 주석 제거
            val = val.split(" #", 1)[0].rstrip()
        if val:                         # 빈 값은 기존 환경변수를 덮지 않음
            os.environ[key] = val
    log.debug(".env 로드: %s", path)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise PipelineError(f"설정 파일이 없습니다: {path}")
    load_dotenv(path.absolute().parent / ".env")
    try:
        with path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise PipelineError(f"config.yaml 문법 오류: {e}") from e
    cfg["_root"] = str(path.absolute().parent)
    validate_config(cfg)
    return cfg


def _num(cfg: dict, dotted: str, errors: list[str], *, positive=True, integer=False):
    cur = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            errors.append(f"{dotted}: 값이 없습니다")
            return None
        cur = cur[k]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)) or (integer and int(cur) != cur):
        errors.append(f"{dotted}: 숫자{'(정수)' if integer else ''}여야 합니다 (현재 {cur!r})")
        return None
    if positive and cur <= 0:
        errors.append(f"{dotted}: 0보다 커야 합니다 (현재 {cur})")
        return None
    return cur


def validate_config(cfg: dict) -> None:
    """잘못된 숫자 설정을 렌더 전에 막음 (문제 필드 이름과 값을 알려 줌)."""
    errors: list[str] = []
    for key in ("video", "layout", "title", "subtitle", "sources", "llm"):
        if not isinstance(cfg.get(key), dict):
            errors.append(f"{key}: 섹션이 없습니다")
    if errors:
        raise PipelineError("config.yaml 오류:\n- " + "\n- ".join(errors))
    w = _num(cfg, "video.width", errors, integer=True)
    h = _num(cfg, "video.height", errors, integer=True)
    _num(cfg, "video.fps", errors, integer=True)
    t = _num(cfg, "video.target_duration", errors)
    m = _num(cfg, "video.max_duration", errors)
    if t and m and t > m:
        errors.append(f"video.target_duration({t}) > video.max_duration({m})")
    for part in ("title", "subtitle"):
        _num(cfg, f"{part}.size", errors, integer=True)
    a = _num(cfg, "subtitle.min_sec", errors)
    b = _num(cfg, "subtitle.max_sec", errors)
    if a and b and a > b:
        errors.append(f"subtitle.min_sec({a}) > subtitle.max_sec({b})")
    lay = cfg["layout"]
    for name, p in (lay.get("presets") or {}).items():
        if not isinstance(p, dict):
            errors.append(f"layout.presets.{name}: 객체여야 합니다")
            continue
        th, vh = p.get("title_h"), p.get("video_h")
        if not all(isinstance(x, int) and x >= 0 for x in (th, vh)) or not vh:
            errors.append(f"layout.presets.{name}: title_h/video_h는 0 이상 정수여야 합니다 ({th}, {vh})")
        elif h and th + vh >= h:
            errors.append(f"layout.presets.{name}: 제목 {th} + 영상 {vh} ≥ 화면 높이 {h} (캔버스 밖, 자막 영역 없음)")
    if lay.get("presets") and lay.get("preset") and lay["preset"] not in lay["presets"]:
        errors.append(f"layout.preset: '{lay['preset']}' 프리셋이 없습니다")
    for k in ("face_fill", "max_upscale"):
        if k in lay:
            _num(cfg, f"layout.{k}", errors)
    if w and w % 2 or h and h % 2:
        errors.append("video.width/height는 짝수여야 합니다 (H.264)")
    if errors:
        raise PipelineError("config.yaml 오류:\n- " + "\n- ".join(errors))


def config_snapshot(cfg: dict) -> dict:
    """작업 폴더에 보관할 설정 사본. 비밀 값은 config가 아니라 .env/환경변수에만 있으므로 포함되지 않음."""
    import copy
    snap = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    for k in list((snap.get("llm") or {})):
        if any(s in k.lower() for s in ("key", "token", "secret")):
            snap["llm"].pop(k)
    return snap


def config_version(path: str | Path) -> str:
    """설정 파일 내용 해시 앞 8자리 (UI에 '적용된 설정 버전'으로 표시)."""
    import hashlib
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:8]


def resolve(cfg: dict, p: str | Path) -> Path:
    """config.yaml 위치 기준 상대경로 해석."""
    p = Path(p)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def require_ffmpeg() -> None:
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise PipelineError(f"{exe}가 PATH에 없습니다. https://ffmpeg.org 에서 설치하세요.")


def run_ffmpeg(args: list[str], what: str) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise PipelineError(f"ffmpeg 실패 ({what}):\n{proc.stderr.strip()[-2000:]}")


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise PipelineError(f"영상 길이를 읽을 수 없습니다: {path}") from e


def slugify(text: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in text)
    return "_".join(filter(None, keep.split("_")))[:40] or "untitled"
