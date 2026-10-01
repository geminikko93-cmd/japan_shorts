"""② 소스 수집: 유튜브 재팬 + 크리에이티브 커먼즈 필터 → 라이선스 재확인 → 다운로드.

영상의 수동 작업(설정 → 국가 '일본' → 필터 '크리에이티브 커먼즈')을 YouTube Data API
`search.list(regionCode=JP, videoLicense=creativeCommon)`로 대체합니다.
다운로드 전 `videos.list(status.license)`로 CC BY인지 한 번 더 검증하고,
출처 표기(attribution)용 메타데이터를 함께 보관합니다.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .common import PipelineError, log

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
BLOCK_LIMIT = 3   # 403/로그인 요구가 연속 이만큼 나오면 다운로드 중단


@dataclass
class Source:
    path: str
    video_id: str = ""
    title: str = ""
    channel: str = ""
    url: str = ""
    license: str = "own"     # creativeCommon | own

    def attribution(self) -> str:
        if self.license == "creativeCommon":
            return f"「{self.title}」 by {self.channel} (CC BY) {self.url}"
        return ""


def _iso_seconds(d: str) -> int:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m:
        return 0
    dd, h, mi, s = (int(x or 0) for x in m.groups())
    return dd * 86400 + h * 3600 + mi * 60 + s


def _youtube():
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise PipelineError(".env에 YOUTUBE_API_KEY가 필요합니다 (Google Cloud Console → YouTube Data API v3).")
    from googleapiclient.discovery import build
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


def search_cc_videos(keywords: list[str], cfg: dict) -> list[dict]:
    """CC 라이선스 영상 후보 검색. search.list는 호출당 100 quota units."""
    from googleapiclient.errors import HttpError

    sc = cfg["sources"]
    yt = _youtube()
    seen, ids = set(), []
    for q in keywords:
        try:
            res = yt.search().list(
                q=q, part="id", type="video", maxResults=15,
                videoLicense="creativeCommon" if sc["require_cc"] else "any",
                regionCode=sc["region"], relevanceLanguage=sc["language"],
                safeSearch="strict",
            ).execute()
        except HttpError as e:
            raise PipelineError(f"YouTube 검색 실패 ({q}): {e}") from e
        for item in res.get("items", []):
            vid = item["id"]["videoId"]
            if vid not in seen:
                seen.add(vid)
                ids.append(vid)
    if not ids:
        return []

    # 라이선스 재검증 + 메타데이터 (videos.list는 1 unit, 50개씩)
    verified = []
    for i in range(0, len(ids), 50):
        res = yt.videos().list(part="snippet,status,contentDetails",
                               id=",".join(ids[i:i + 50])).execute()
        for v in res.get("items", []):
            if sc["require_cc"] and v["status"].get("license") != "creativeCommon":
                continue
            if not v["status"].get("embeddable", True):
                continue
            verified.append({
                "video_id": v["id"],
                "title": v["snippet"]["title"],
                "channel": v["snippet"]["channelTitle"],
                "url": f"https://www.youtube.com/watch?v={v['id']}",
                "license": v["status"].get("license", ""),
                "thumbnail": v["snippet"].get("thumbnails", {}).get("medium", {}).get("url", ""),
                "seconds": _iso_seconds(v["contentDetails"].get("duration", "")),
            })
    log.info("CC 후보 %d개 (검색 %d개 중 라이선스 검증 통과)", len(verified), len(ids))
    return verified


def rank_for_auto(candidates: list[dict], names: list[str]) -> list[dict]:
    """자동 제작용 후보 순서. 제목에 인물 이름이 있고, 너무 짧거나 길지 않은 영상 우선 (검색 순서 유지)."""
    names = [n.replace(" ", "") for n in names if n]

    def score(c: dict) -> int:
        title, secs = c["title"].replace(" ", ""), c.get("seconds", 0)
        s = 2 if any(n in title for n in names) else 0
        if 30 <= secs <= 20 * 60:      # 인터뷰·제작발표회·MV 길이
            s += 1
        elif secs > 60 * 60:           # 1시간 넘으면 다운로드가 무거움
            s -= 2
        return s

    return sorted(candidates, key=score, reverse=True)   # sorted는 안정 정렬 → 동점이면 검색 순위


def download(candidates: list[dict], out_dir: Path, cfg: dict) -> list[Source]:
    import yt_dlp

    sc = cfg["sources"]
    out_dir.mkdir(parents=True, exist_ok=True)
    h = sc["max_download_height"]
    opts = {
        # H.264(avc1) 우선: AV1/VP9는 OpenCV 탐색이 수십 배 느림
        "format": (f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/bv*[height<={h}][ext=mp4]+ba[ext=m4a]/"
                   f"b[height<={h}][ext=mp4]/b[height<={h}]"),
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True, "no_warnings": True, "noplaylist": True,
    }
    sources: list[Source] = []
    blocked_streak = 0
    with yt_dlp.YoutubeDL(opts) as ydl:
        for c in candidates:
            if len(sources) >= sc["max_videos"]:
                break
            meta = {k: c[k] for k in ("video_id", "title", "channel", "url", "license")}
            cached = out_dir / f"{c['video_id']}.mp4"
            if cached.exists():   # 이전 실행에서 받은 영상 재사용 → YouTube 요청 최소화 (먼저 검사)
                problem = media_problem(cached)
                if problem is None:
                    sources.append(Source(path=str(cached), **meta))
                    log.info("이미 받은 영상 사용: %s", cached.name)
                    continue
                log.warning("캐시 영상이 손상되어 다시 받습니다: %s (%s)", cached.name, problem)
                cached.unlink(missing_ok=True)
            try:
                info = ydl.extract_info(c["url"], download=False)
                lic = (info.get("license") or "").lower()
                if sc["require_cc"] and lic and "creative commons" not in lic:
                    log.warning("라이선스 불일치로 건너뜀: %s (%s)", c["url"], lic)
                    continue
                info = ydl.extract_info(c["url"], download=True)
                path = _downloaded_file(ydl, info, out_dir)
                problem = media_problem(path)
                if problem:
                    path.unlink(missing_ok=True)
                    raise RuntimeError(f"받은 파일이 영상으로 읽히지 않음: {problem}")
            except Exception as e:  # yt-dlp는 다양한 예외를 던짐 → 해당 영상만 건너뜀
                msg = str(e)
                for part in out_dir.glob(f"{c['video_id']}*.part"):
                    part.unlink(missing_ok=True)
                if "403" in msg or "sign in" in msg.lower():
                    blocked_streak += 1
                    log.warning("YouTube가 다운로드를 차단함 (%d/%d): %s",
                                blocked_streak, BLOCK_LIMIT, c["url"])
                    if blocked_streak >= BLOCK_LIMIT:
                        log.warning("연속 차단 → 다운로드 중단 (계속 시도하면 IP 차단이 심해질 수 있음)")
                        break
                else:
                    log.warning("다운로드 실패, 건너뜀: %s (%s)", c["url"], msg)
                continue
            blocked_streak = 0
            sources.append(Source(path=str(path), **meta))
            log.info("다운로드: %s", path.name)
    if not sources:
        raise PipelineError(
            "CC 영상을 받지 못했습니다. YouTube가 다운로드를 막고 있다면 사용 권한이 있는 영상을 "
            "폴더에 모아 --sources-dir 로 지정하세요.")
    if len(sources) < sc["max_videos"]:
        log.warning("소스 %d개만 확보 (목표 %d개) → 확보한 영상으로 진행", len(sources), sc["max_videos"])
    return sources


def _downloaded_file(ydl, info: dict, out_dir: Path) -> Path:
    """yt-dlp가 실제로 만든 파일 (병합 결과 우선). 임의 파일을 고르지 않는다."""
    cands = []
    for d in info.get("requested_downloads") or []:
        if d.get("filepath"):
            cands.append(Path(d["filepath"]))
    base = Path(ydl.prepare_filename(info))
    cands += [base.with_suffix(".mp4"), base]
    for p in cands:
        if p.exists() and p.suffix.lower() in VIDEO_EXTS and p.parent.resolve() == out_dir.resolve():
            return p
    raise RuntimeError(f"다운로드 결과 파일을 찾지 못함 ({info.get('id')})")


def media_problem(path: Path) -> str | None:
    """ffprobe로 영상 스트림·길이·해상도 확인. 문제 없으면 None."""
    import subprocess
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=width,height:format=duration", "-of", "json", str(path)],
                              capture_output=True, text=True, timeout=60)
        data = json.loads(proc.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return f"ffprobe 실패: {e}"
    st = (data.get("streams") or [{}])[0]
    try:
        dur = float((data.get("format") or {}).get("duration", 0))
    except ValueError:
        dur = 0.0
    if not st.get("width") or not st.get("height"):
        return "비디오 스트림 없음"
    if dur < 1.0:
        return f"길이가 너무 짧음 ({dur:.2f}초)"
    return None


def from_local_dir(d: Path) -> list[Source]:
    """직접 촬영/보유/라이선스 확보한 영상 폴더 사용.

    sources.json(선택)은 save_sources()가 쓰는 형식과 같다: [{file 또는 path, title, channel, url, license, ...}]
    """
    if not d.is_dir():
        raise PipelineError(f"소스 폴더가 없습니다: {d}")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise PipelineError(f"소스 폴더에 영상이 없습니다: {d}")
    meta_file = d / "sources.json"   # 선택: 출처 정보 [{file, title, channel, url, license}]
    meta = {}
    if meta_file.exists():
        try:
            items = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise PipelineError(f"{meta_file} 형식 오류: {e}") from e
        for i, m in enumerate(items):
            name = m.get("file") or (Path(m["path"]).name if m.get("path") else "")
            if not name:
                raise PipelineError(f"{meta_file}[{i}]: file 또는 path가 필요합니다")
            meta[name] = m
    out = []
    for f in files:
        m = meta.get(f.name, {})
        out.append(Source(path=str(f), video_id=m.get("video_id", ""), title=m.get("title", f.stem),
                          channel=m.get("channel", ""), url=m.get("url", ""), license=m.get("license", "own")))
    return out


def save_sources(sources: list[Source], path: Path) -> None:
    """from_local_dir()가 다시 읽을 수 있는 형식 (file = 파일 이름, path = 전체 경로)."""
    path.write_text(json.dumps([{"file": Path(s.path).name, **asdict(s)} for s in sources],
                               ensure_ascii=False, indent=2), encoding="utf-8")
