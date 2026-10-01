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
import time
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


def yt_error(e, what: str) -> PipelineError:
    """googleapiclient HttpError → 사용자용 오류. URL에 들어 있는 API 키는 절대 그대로 보여 주지 않는다."""
    status = getattr(getattr(e, "resp", None), "status", "?")
    reason, msg = "", ""
    try:
        err = json.loads(e.content.decode("utf-8"))["error"]
        msg = err.get("message", "")
        reason = (err.get("errors") or [{}])[0].get("reason", "")
    except (AttributeError, ValueError, KeyError, TypeError):
        msg = str(e)
    msg = re.sub(r"key=[A-Za-z0-9_\-]+", "key=***", msg)
    if str(status) == "429" or reason in ("quotaExceeded", "rateLimitExceeded", "dailyLimitExceeded"):
        return PipelineError(f"{what}: YouTube API 일일 할당량(또는 하루 검색 횟수)을 다 썼습니다. 태평양 시간 자정"
                             "(한국 시간 오후 4~5시)에 초기화됩니다. 이미 받은 영상·저장된 검색 결과·내 영상 폴더는 계속 쓸 수 있습니다.")
    if reason in ("keyInvalid", "badRequest") and "key" in msg.lower():
        return PipelineError(f"{what}: YouTube API 키가 올바르지 않습니다 (.env의 YOUTUBE_API_KEY 확인).")
    return PipelineError(f"{what}: YouTube API 오류 {status} {reason} {msg[:200]}".strip())


def _cache_path(cfg: dict) -> Path:
    from .common import resolve
    return resolve(cfg, cfg.get("work_dir", "work")) / "cache" / "cc_search.json"


def _cache_load(cfg: dict) -> dict:
    p = _cache_path(cfg)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _cache_save(cfg: dict, cache: dict) -> None:
    p = _cache_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 7 * 86400
    cache = {k: v for k, v in cache.items() if v.get("t", 0) >= cutoff}
    p.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _youtube():
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise PipelineError(".env에 YOUTUBE_API_KEY가 필요합니다 (Google Cloud Console → YouTube Data API v3).")
    from googleapiclient.discovery import build
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


# ───────────────────────── CC 검색 (키워드 확장 + 여러 페이지) ─────────────────────────
EVENT_SUFFIXES = ["イベント", "舞台挨拶", "記者会見", "インタビュー", "フォトコール", "制作発表"]
EVENT_WORDS = EVENT_SUFFIXES + ["会見", "挨拶", "試写会", "発表会", "登壇", "レッドカーペット", "photocall", "photo call",
                                "red carpet", "premiere", "press", "포토콜", "제작발표회", "레드카펫", "시사회", "행사",
                                "인터뷰", "기자회견", "포토월", "팬미팅"]
LOOKALIKE_WORDS = ["激似", "そっくり", "ものまね", "モノマネ", "似てる", "似すぎ", "似ている", "に似", "風メイク", "닮은",
                   "닮았", "도플갱어", "따라하기", "lookalike", "look alike", "impersonat"]
COMPILATION_WORDS = ["プロフィール", "経歴", "まとめ", "ランキング", "スライド", "画像集", "写真集", "生い立ち", "歴代",
                     "wiki", "比較", "変遷", "모음", "총정리", "랭킹", "프로필", "slideshow", "compilation"]


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        x = (x or "").strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def expand_keywords(script: dict, extra: list[str] | None = None, max_n: int | None = None,
                    cfg: dict | None = None) -> list[str]:
    """CC 검색어: 사용자 추가어 → 이름 → 이름+행사어 → 한국어 이름 → 기획안 검색어 (중복 제거, 최대 max_n개).

    CC 영상은 일본 방송국보다 한국 언론·행사 채널이 올린 것이 많아 한국어 이름과 행사어를 함께 쓴다.
    """
    name_ja = (script.get("name_ja") or "").strip()
    name_ko = (script.get("name_ko") or "").strip()
    max_n = max_n or int((cfg or {}).get("sources", {}).get("max_keywords", 6))
    kws = list(extra or [])
    if name_ja:
        kws += [name_ja, f"{name_ja} {EVENT_SUFFIXES[0]}", f"{name_ja} {EVENT_SUFFIXES[1]}"]
    if name_ko:
        kws += [name_ko, f"{name_ko} 포토콜"]
    if name_ja:
        kws += [f"{name_ja} {x}" for x in EVENT_SUFFIXES[2:]]
    kws += list(script.get("search_keywords_ja") or [])
    return _dedupe(kws)[:max_n]


def estimate_units(n_keywords: int, pages: int, expected_results: int | None = None) -> int:
    """search.list 1회 = 100 units, videos.list 50개당 1 unit."""
    n = expected_results if expected_results is not None else n_keywords * pages * 50
    return n_keywords * pages * 100 + (n + 49) // 50


def _flat(text: str) -> str:
    return text.replace(" ", "").replace("　", "").lower()


def assess(c: dict, names: list[str]) -> dict:
    """다운로드 전, 제목·설명·길이만으로 '쓸 만해 보이는지' 대략 판단 (영상 내용은 보지 않음 → 받은 뒤 분석이 최종).

    반환: {"score": 정수, "flags": [표시용 문구], "likely": bool}
    """
    flat = _flat(f"{c.get('title', '')} {c.get('description', '')[:300]}")
    title_flat = _flat(c.get("title", ""))
    names = [_flat(n) for n in names if n]
    secs = int(c.get("seconds", 0) or 0)
    score, flags = 0, []
    if any(n in title_flat for n in names):
        score += 3
    elif any(n in flat for n in names):
        score += 1
        flags.append("이름이 설명에만 있음")
    else:
        flags.append("제목·설명에 이름 없음")
    if any(_flat(w) in flat for w in LOOKALIKE_WORDS):
        score -= 6
        flags.append("닮은 사람 영상일 수 있음")
    if any(_flat(w) in flat for w in COMPILATION_WORDS):
        score -= 3
        flags.append("사진·정리 영상일 수 있음")
    if any(_flat(w) in flat for w in EVENT_WORDS):
        score += 2
        flags.append("행사·인터뷰 영상")
    if secs and secs < 20:
        score -= 1
        flags.append("20초 미만")
    elif 30 <= secs <= 20 * 60:
        score += 1
    elif secs > 60 * 60:
        score -= 2
        flags.append("1시간 넘음 (다운로드·분석 오래 걸림)")
    if c.get("definition") == "hd":
        score += 1
    elif c.get("definition"):
        flags.append("SD 화질")
    likely = score >= 3 and "닮은 사람 영상일 수 있음" not in flags
    return {"score": score, "flags": flags, "likely": likely}


def search_cc_videos(keywords: list[str], cfg: dict, pages: int | None = None, names: list[str] | None = None,
                     yt=None, stats: dict | None = None, use_cache: bool = True) -> list[dict]:
    """CC 라이선스 영상 후보 검색 → videos.list로 라이선스 재검증 → 메타데이터 기반 사전 판단(assess).

    search.list는 호출당 100 quota units (키워드 × 페이지). 결과는 '쓸 만해 보이는' 순서(동점이면 검색 순).
    같은 검색(검색어·설정·페이지)은 sources.search_cache_hours 동안 저장된 결과를 다시 써서 할당량을 아낀다.
    stats에 실제 API 검색 횟수(api_searches)와 캐시 사용 횟수(cached), 추정 units를 기록한다.
    """
    import hashlib

    from googleapiclient.errors import HttpError

    sc = cfg["sources"]
    pages = max(int(pages or sc.get("search_pages", 1)), 1)
    stats = stats if stats is not None else {}
    stats.setdefault("api_searches", 0)
    stats.setdefault("cached", 0)
    ttl = float(sc.get("search_cache_hours", 24)) * 3600
    cache = _cache_load(cfg) if use_cache else {}
    seen, ids, hits = set(), [], {}
    for q in keywords:
        token = None
        for _ in range(pages):
            params = dict(q=q, part="id", type="video", maxResults=50,
                          videoLicense="creativeCommon" if sc["require_cc"] else "any", safeSearch="strict")
            if sc.get("region"):
                params["regionCode"] = sc["region"]
            if sc.get("relevance_language"):
                params["relevanceLanguage"] = sc["relevance_language"]
            if token:
                params["pageToken"] = token
            ck = hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            hit = cache.get(ck)
            if hit and time.time() - hit.get("t", 0) < ttl:
                res = hit["res"]
                stats["cached"] += 1
            else:
                yt = yt or _youtube()
                try:
                    res = yt.search().list(**params).execute()
                except HttpError as e:
                    if use_cache:
                        _cache_save(cfg, cache)          # 이미 받은 페이지는 저장 → 다음에 할당량을 안 씀
                    raise yt_error(e, f"YouTube 검색 '{q}'") from e
                stats["api_searches"] += 1
                res = {"items": res.get("items", []), "nextPageToken": res.get("nextPageToken")}
                cache[ck] = {"t": time.time(), "res": res}
            for item in res.get("items", []):
                vid = (item.get("id") or {}).get("videoId")
                if not vid:                      # 채널·재생목록 등 영상이 아닌 결과가 섞여 올 때가 있음
                    continue
                hits.setdefault(vid, []).append(q)
                if vid not in seen:
                    seen.add(vid)
                    ids.append(vid)
            token = res.get("nextPageToken")
            if not token:
                break
    if use_cache:
        _cache_save(cfg, cache)
    if not ids:
        return []

    # 라이선스 재검증 + 메타데이터 (videos.list는 1 unit, 50개씩 — 라이선스가 바뀔 수 있어 캐시하지 않음)
    verified = []
    names = names or keywords[:1]
    yt = yt or _youtube()
    for i in range(0, len(ids), 50):
        try:
            res = yt.videos().list(part="snippet,status,contentDetails", id=",".join(ids[i:i + 50])).execute()
        except HttpError as e:
            raise yt_error(e, "YouTube 영상 정보 조회") from e
        stats["video_lists"] = stats.get("video_lists", 0) + 1
        for v in res.get("items", []):
            if sc["require_cc"] and v["status"].get("license") != "creativeCommon":
                continue
            if not v["status"].get("embeddable", True):
                continue
            c = {
                "video_id": v["id"],
                "title": v["snippet"]["title"],
                "description": v["snippet"].get("description", "")[:500],
                "channel": v["snippet"]["channelTitle"],
                "url": f"https://www.youtube.com/watch?v={v['id']}",
                "license": v["status"].get("license", ""),
                "thumbnail": v["snippet"].get("thumbnails", {}).get("medium", {}).get("url", ""),
                "seconds": _iso_seconds(v["contentDetails"].get("duration", "")),
                "definition": v["contentDetails"].get("definition", ""),
                "keywords": hits.get(v["id"], []),
            }
            c["assess"] = assess(c, names)
            verified.append(c)
    order = {vid: k for k, vid in enumerate(ids)}
    verified.sort(key=lambda c: (-c["assess"]["score"], order[c["video_id"]]))
    likely = sum(c["assess"]["likely"] for c in verified)
    stats["units"] = stats["api_searches"] * 100 + stats.get("video_lists", 0)
    log.info("CC 후보 %d개 (검색 %d개 중 라이선스 검증 통과), 쓸 만해 보이는 후보 %d개 — 키워드 %d개 × %d페이지, "
             "API 검색 %d회(저장된 결과 %d회), 약 %d units", len(verified), len(ids), likely, len(keywords), pages,
             stats["api_searches"], stats["cached"], stats["units"])
    return verified


def rank_for_auto(candidates: list[dict], names: list[str]) -> list[dict]:
    """자동 제작용 후보 순서: 사전 판단 점수 높은 순, 닮은 사람 영상 의심은 맨 뒤 (동점이면 검색 순위)."""
    scored = [(c, c.get("assess") or assess(c, names)) for c in candidates]
    scored.sort(key=lambda x: ("닮은 사람 영상일 수 있음" in x[1]["flags"], -x[1]["score"]))
    return [c for c, _ in scored]


def check_person(script_like: dict, cfg: dict, max_keywords: int = 3, yt=None) -> dict:
    """인물별 CC 소스 미리 확인 (다운로드 없음). 키워드 max_keywords개 × 1페이지 = 약 100×개 units."""
    kws = expand_keywords(script_like, max_n=max_keywords)
    stats: dict = {}
    found = search_cc_videos(kws, cfg, pages=1, names=[script_like.get("name_ja", ""), script_like.get("name_ko", "")],
                             yt=yt, stats=stats)
    likely = [c for c in found if c["assess"]["likely"]]
    top = []
    for c in (likely or found)[:5]:
        top.append({k: c[k] for k in ("title", "url", "channel", "seconds", "thumbnail")} | {"flags": c["assess"]["flags"]})
    return {"checked": time.strftime("%Y-%m-%d %H:%M"), "keywords": kws, "total": len(found), "likely": len(likely),
            "units": stats.get("units", 0), "cached": stats.get("cached", 0), "top": top}


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
