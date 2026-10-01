"""⑦ 결과물: 업로드용 메타데이터(제목/설명/해시태그/출처) + 선택적 YouTube 업로드.

업로드는 기본 '비공개(private)'로 올립니다 → 사람이 확인 후 공개 전환 (휴먼 인 더 루프).
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import PipelineError, log
from .sources import Source

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def build_metadata(script: dict, title_idx: int, sources: list[Source], music_credit: str) -> dict:
    t = script["titles"][title_idx]
    tags = [h if h.startswith("#") else f"#{h}" for h in script.get("hashtags", [])]
    if "#shorts" not in [x.lower() for x in tags]:
        tags.insert(0, "#shorts")
    yt_title = f"{t['line1']}{t['line2']}"[:90] + " #shorts"
    credits = [s.attribution() for s in sources if s.attribution()]
    desc = [f"{t['line1']}{t['line2']}", "", " ".join(tags), ""]
    if credits:
        desc += ["▼ 使用素材 (Creative Commons)", *credits, ""]
    if music_credit:
        desc += [f"♪ {music_credit}", ""]
    desc.append("※ファンによる応援・紹介動画です。")
    return {"title": yt_title, "description": "\n".join(desc),
            "tags": [x.lstrip("#") for x in tags], "title_ko": t["ko"]}


def save_metadata(meta: dict, out_dir: Path) -> None:
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "description.txt").write_text(meta["description"], encoding="utf-8")


def upload(video: Path, meta: dict, client_secret: Path, token_file: Path, privacy: str = "private") -> str:
    """YouTube Data API videos.insert (1,600 quota units/회)."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not client_secret.exists():
                raise PipelineError(f"OAuth client_secret 파일이 없습니다: {client_secret}")
            creds = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES).run_local_server(port=0)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = {
        "snippet": {"title": meta["title"], "description": meta["description"],
                    "tags": meta["tags"][:15], "categoryId": "24", "defaultLanguage": "ja"},
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }
    try:
        req = yt.videos().insert(part="snippet,status", body=body,
                                 media_body=MediaFileUpload(str(video), resumable=True))
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
    except HttpError as e:
        raise PipelineError(f"업로드 실패: {e}") from e
    log.info("업로드 완료(%s): https://youtube.com/shorts/%s", privacy, resp["id"])
    return resp["id"]
