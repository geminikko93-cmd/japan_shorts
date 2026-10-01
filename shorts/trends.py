"""⓪ 트렌드 참고: 최근 N시간 안에 '업로드된' 일본 짧은 영상의 업로드 이후 시간당 평균 조회수.

주의: avg_views_per_hour_since_upload = 총조회수 / max(업로드 후 경과 시간, 0.5시간).
최근 1시간·24시간 동안 실제로 늘어난 조회수가 아니며, 검색 기간에 업로드된 영상만 대상이라
오래된 영상의 재유행은 잡지 못한다. 길이 필터만 통과한 영상은 '짧은 영상 후보'이지 Shorts 확정이 아니다.

영상 속 내부 프로그램('샤르르르')과 같은 목적을 YouTube Data API로 구현:
- search.list(publishedAfter=24h, regionCode=JP, videoDuration=short, order=viewCount)
- videos.list(statistics) → 업로드 이후 시간당 평균 조회수, 좋아요율, 추정 수익(RPM 기반)
Quota: search 1회 = 100 units (기본 일일 10,000 units) — 키워드 수를 아껴 쓰세요.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .common import PipelineError, log
from .sources import _iso_seconds, _youtube

CATEGORY_NAMES = {
    "1": "영화/애니", "2": "자동차", "10": "음악", "15": "동물", "17": "스포츠", "19": "여행",
    "20": "게임", "22": "인물/블로그", "23": "코미디", "24": "엔터테인먼트", "25": "뉴스",
    "26": "노하우/스타일", "27": "교육", "28": "과학기술",
}


def scan(cfg: dict, queries: list[str], out_csv: Path) -> list[dict]:
    from googleapiclient.errors import HttpError

    tc = cfg["trends"]
    yt = _youtube()
    since = datetime.now(timezone.utc) - timedelta(hours=tc["hours"])
    ids: list[str] = []
    for q in queries or ["#shorts"]:
        try:
            res = yt.search().list(
                q=q, part="id", type="video", order="viewCount", videoDuration="short",
                regionCode=tc["region"], relevanceLanguage="ja",
                publishedAfter=since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                maxResults=min(tc["max_results"], 50),
            ).execute()
        except HttpError as e:
            raise PipelineError(f"트렌드 검색 실패 ({q}): {e}") from e
        ids += [it["id"]["videoId"] for it in res.get("items", []) if it["id"]["videoId"] not in ids]

    rows = []
    now = datetime.now(timezone.utc)
    for i in range(0, len(ids), 50):
        res = yt.videos().list(part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50])).execute()
        for v in res.get("items", []):
            secs = _iso_seconds(v["contentDetails"].get("duration", ""))
            if secs == 0 or secs > tc["max_shorts_seconds"]:
                continue
            st, sn = v.get("statistics", {}), v["snippet"]
            views = int(st.get("viewCount", 0))
            likes = int(st.get("likeCount", 0))
            pub = datetime.fromisoformat(sn["publishedAt"].replace("Z", "+00:00"))
            hours = max((now - pub).total_seconds() / 3600, 0.5)
            rows.append({
                "title": sn["title"], "channel": sn["channelTitle"],
                "category": CATEGORY_NAMES.get(sn.get("categoryId", ""), sn.get("categoryId", "")),
                "seconds": secs, "short_candidate": True, "views": views,
                "avg_views_per_hour_since_upload": round(views / hours),
                "like_rate_%": round(likes / views * 100, 2) if views else 0,
                "comments": int(st.get("commentCount", 0)),
                "est_revenue_usd": round(views / 1000 * tc["rpm_usd"], 2),
                "hours_since_upload": round(hours, 1),
                "url": f"https://www.youtube.com/shorts/{v['id']}",
            })
    rows.sort(key=lambda r: -r["avg_views_per_hour_since_upload"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:   # 엑셀 한글 깨짐 방지
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["views"]
    log.info("트렌드 %d개 저장 → %s", len(rows), out_csv)
    for cat, views in sorted(by_cat.items(), key=lambda x: -x[1])[:5]:
        log.info("  카테고리 %-10s 합계 조회수 %s", cat, f"{views:,}")
    return rows
