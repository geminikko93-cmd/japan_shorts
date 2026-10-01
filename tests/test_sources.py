"""CC 검색 확장·사전 판단·캐시·오류 처리 테스트 (가짜 YouTube 클라이언트, 실제 API 호출 없음)."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shorts import sources  # noqa: E402
from shorts.common import PipelineError  # noqa: E402

SECRET = "AIzaSyFAKEKEY1234567890"


class FakeHttpError(Exception):
    pass


def http_error(status: int, reason: str, message: str):
    from googleapiclient.errors import HttpError

    class Resp(dict):
        def __init__(self, status):
            super().__init__()
            self.status = status
            self.reason = reason

    body = json.dumps({"error": {"code": status, "message": message, "errors": [{"reason": reason}]}}).encode()
    return HttpError(Resp(status), body, uri=f"https://youtube.googleapis.com/youtube/v3/search?q=x&key={SECRET}")


class FakeYT:
    """search().list(**p).execute() / videos().list(**p).execute() 흉내."""

    def __init__(self, pages: dict, meta: dict, fail_on: str | None = None):
        self.pages, self.meta, self.fail_on = pages, meta, fail_on
        self.search_calls, self.video_calls = [], 0

    def search(self):
        yt = self

        class S:
            def list(self, **p):
                class R:
                    def execute(self):
                        yt.search_calls.append(p)
                        if yt.fail_on and p["q"] == yt.fail_on:
                            raise http_error(429, "rateLimitExceeded", "Quota exceeded for quota metric 'Search Queries'")
                        return yt.pages[(p["q"], p.get("pageToken"))]
                return R()
        return S()

    def videos(self):
        yt = self

        class V:
            def list(self, **p):
                class R:
                    def execute(self):
                        yt.video_calls += 1
                        return {"items": [yt.meta[i] for i in p["id"].split(",") if i in yt.meta]}
                return R()
        return V()


def meta(vid, title, secs=120, lic="creativeCommon", hd="hd", desc=""):
    m, s = divmod(secs, 60)
    return {"id": vid, "snippet": {"title": title, "channelTitle": "ch", "description": desc, "thumbnails": {}},
            "status": {"license": lic, "embeddable": True},
            "contentDetails": {"duration": f"PT{m}M{s}S", "definition": hd}}


class SourceSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cfg = {"_root": str(self.tmp), "work_dir": "work",
                    "sources": {"require_cc": True, "region": "JP", "relevance_language": "", "search_pages": 2,
                                "max_keywords": 6, "search_cache_hours": 24}}
        self.pages = {
            ("本田翼", None): {"items": [{"id": {"videoId": "a"}}, {"id": {"channelId": "zz"}}, {"id": {"videoId": "b"}}],
                             "nextPageToken": "P2"},
            ("本田翼", "P2"): {"items": [{"id": {"videoId": "c"}}]},
            ("혼다 츠바사", None): {"items": [{"id": {"videoId": "a"}}, {"id": {"videoId": "d"}}, {"id": {"videoId": "e"}}]},
        }
        self.meta = {
            "a": meta("a", "本田翼 フォトコール CLANE", 117),
            "b": meta("b", "本田翼さんに激似で駅前がパニックに", 690),
            "c": meta("c", "本田翼 プロフィール まとめ", 170, hd="sd"),
            "d": meta("d", "혼다 츠바사 포토콜 현장", 60),
            "e": meta("e", "다른 영상", 60, lic="youtube"),
        }

    def test_expand_keywords(self):
        kws = sources.expand_keywords({"name_ja": "本田翼", "name_ko": "혼다 츠바사", "search_keywords_ja": ["本田翼"]},
                                      extra=["Tsubasa Honda"], max_n=6)
        self.assertEqual(kws, ["Tsubasa Honda", "本田翼", "本田翼 イベント", "本田翼 舞台挨拶", "혼다 츠바사", "혼다 츠바사 포토콜"])

    def test_pagination_missing_videoid_license_and_ranking(self):
        yt = FakeYT(self.pages, self.meta)
        stats = {}
        got = sources.search_cc_videos(["本田翼", "혼다 츠바사"], self.cfg, names=["本田翼", "혼다 츠바사"], yt=yt, stats=stats)
        self.assertEqual(len(yt.search_calls), 3)                       # 2페이지 + 1페이지
        self.assertNotIn("relevanceLanguage", yt.search_calls[0])       # 비우면 언어 가중치 없음
        self.assertEqual(sorted(c["video_id"] for c in got), ["a", "b", "c", "d"])   # e는 CC 아님, zz는 채널
        by = {c["video_id"]: c for c in got}
        self.assertTrue(by["a"]["assess"]["likely"])
        self.assertIn("닮은 사람 영상일 수 있음", by["b"]["assess"]["flags"])
        self.assertFalse(by["b"]["assess"]["likely"])
        self.assertIn("사진·정리 영상일 수 있음", by["c"]["assess"]["flags"])
        self.assertEqual(got[-1]["video_id"], "b")                      # 닮은 사람 의심은 맨 뒤
        self.assertEqual(by["a"]["keywords"], ["本田翼", "혼다 츠바사"])
        self.assertEqual(stats["api_searches"], 3)
        self.assertEqual(stats["units"], 301)
        ranked = sources.rank_for_auto(got, ["本田翼"])
        self.assertEqual(ranked[-1]["video_id"], "b")

    def test_cache_avoids_repeated_quota(self):
        yt = FakeYT(self.pages, self.meta)
        sources.search_cc_videos(["本田翼"], self.cfg, yt=yt)
        stats = {}
        sources.search_cc_videos(["本田翼"], self.cfg, yt=yt, stats=stats)
        self.assertEqual(len(yt.search_calls), 2)                       # 두 번째는 API 검색 없음
        self.assertEqual((stats["api_searches"], stats["cached"]), (0, 2))
        self.assertEqual(stats["units"], 1)                             # videos.list만

    def test_quota_error_hides_key_and_keeps_partial_cache(self):
        yt = FakeYT(self.pages, self.meta, fail_on="혼다 츠바사")
        with self.assertRaises(PipelineError) as e:
            sources.search_cc_videos(["本田翼", "혼다 츠바사"], self.cfg, yt=yt)
        msg = str(e.exception)
        self.assertNotIn(SECRET, msg)
        self.assertIn("할당량", msg)
        yt2 = FakeYT(self.pages, self.meta)
        sources.search_cc_videos(["本田翼"], self.cfg, yt=yt2)            # 실패 전에 받은 페이지는 캐시됨
        self.assertEqual(yt2.search_calls, [])

    def test_error_message_never_contains_key(self):
        err = sources.yt_error(http_error(400, "badRequest", f"API key not valid key={SECRET}"), "검색")
        self.assertNotIn(SECRET, str(err))
        err = sources.yt_error(http_error(500, "backendError", f"boom key={SECRET}"), "검색")
        self.assertNotIn(SECRET, str(err))
        self.assertIn("key=***", str(err))

    def test_check_person_summary(self):
        yt = FakeYT({("本田翼", None): self.pages[("本田翼", None)], ("本田翼", "P2"): self.pages[("本田翼", "P2")],
                     ("本田翼 イベント", None): {"items": []}, ("本田翼 舞台挨拶", None): {"items": []}}, self.meta)
        r = sources.check_person({"name_ja": "本田翼", "name_ko": ""}, self.cfg, yt=yt)
        self.assertEqual((r["total"], r["likely"]), (2, 1))                 # 미리 확인은 1페이지만
        self.assertEqual(r["keywords"], ["本田翼", "本田翼 イベント", "本田翼 舞台挨拶"])
        self.assertEqual(r["top"][0]["url"], "https://www.youtube.com/watch?v=a")
        self.assertEqual(r["units"], 301)


if __name__ == "__main__":
    unittest.main(verbosity=2)
