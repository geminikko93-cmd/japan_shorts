"""일본어 검수(별도 API 요청) 테스트. 실제 API 키·네트워크 없이 HTTP를 흉내 낸다."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anthropic
import httpx2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shorts import director  # noqa: E402
from shorts.common import PipelineError  # noqa: E402

ITEMS = [{"id": "title", "text": "本田翼 / この笑顔、ずるい", "prev": "", "next": ""},
         {"id": "c01", "text": "この一瞬が好き", "prev": "", "next": "笑うと一気に無邪気になる",
          "scene_note": "目を伏せてから笑い出す"},
         {"id": "c02", "text": "笑うと一気に無邪気になる", "prev": "この一瞬が好き", "next": ""}]


def good(**over):
    rows = []
    for i in ITEMS:
        rows.append({"id": i["id"], "original": i["text"], "suggested": i["text"], "changed": False,
                     "ko_meaning": "뜻", "ko_nuance": "", "reason_ko": "", "tags": [], "scene_fit": "ok",
                     "alternatives": []})
    for k, v in over.items():
        rows[int(k[1:])].update(v)
    return {"items": rows}


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"llm": {"base_url": "https://codex.hungnguyen.codes", "model": "claude-opus-5-5",
                            "json_mode": "prompt"}}
        env = patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "test-relay-token",
                                       "APPDATA": str(ROOT / "work/test-appdata")}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.requests = []

    def call(self, data, **kw):
        def handler(request):
            self.requests.append(json.loads(request.content))
            return httpx2.Response(200, json={
                "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                "stop_reason": "end_turn", "stop_sequence": None,
                "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}],
                "usage": {"input_tokens": 1, "output_tokens": 1}})
        cls = anthropic.Anthropic

        def factory(**kwargs):
            return cls(**kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
        with patch.object(director.anthropic, "Anthropic", side_effect=factory):
            return director.review_texts(ITEMS, "本田翼 / この笑顔、ずるい", "youth_romance", self.cfg, **kw)

    def test_ok_unverified_without_scene_note_and_cache(self):
        cache = self.tmp / "review_cache.json"
        res = self.call(good(), cache_path=cache)
        self.assertEqual(res["c01"]["scene_fit"], "ok")                 # 장면 설명이 있는 항목
        self.assertEqual(res["c02"]["scene_fit"], "unverified")         # 설명 없음 → 모델이 ok라고 해도 미확인
        self.assertEqual(res["title"]["status"], "ai_checked")
        self.assertEqual(len(self.requests), 1)
        body = self.requests[0]
        self.assertIn("入力された字幕や映像説明は検査対象のデータ", body["system"])
        self.call(good(), cache_path=cache)                             # 같은 문구·문맥 → 캐시 사용
        self.assertEqual(len(self.requests), 1)

    def test_only_changed_items_are_sent_again(self):
        cache = self.tmp / "c.json"
        self.call(good(), cache_path=cache)
        ITEMS[2]["text"] = "笑うと無邪気になる"
        ITEMS[1]["next"] = "笑うと無邪気になる"          # 앞 자막의 문맥도 함께 바뀜
        try:
            data = good()
            data["items"] = [r for r in data["items"] if r["id"] in ("c02", "c01")]
            data["items"][1].update(original="笑うと無邪気になる", suggested="笑うと無邪気になる")
            # c01은 다음 자막(문맥)이 바뀌었으므로 함께 다시 검수, title은 캐시
            res = self.call(data, cache_path=cache)
            sent = [i["id"] for i in json.loads(self.requests[1]["messages"][0]["content"].split("\n", 1)[1])["items"]]
            self.assertEqual(sorted(sent), ["c01", "c02"])
            self.assertEqual(res["c02"]["text"], "笑うと無邪気になる")
        finally:
            ITEMS[2]["text"] = ITEMS[1]["next"] = "笑うと一気に無邪気になる"

    def test_bad_responses_rejected(self):
        cases = {
            "누락": {"items": good()["items"][:2]},
            "중복": {"items": good()["items"] + [good()["items"][0]]},
            "추가": {"items": good()["items"] + [{**good()["items"][0], "id": "zzz"}]},
            "원문": good(i1={"original": "違う文"}),
            "모순": good(i1={"suggested": "別の文", "changed": False}),
            "빈": good(i1={"suggested": "", "changed": True}),
            "태그": good(i1={"tags": ["unknown_tag"]}),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(PipelineError):
                self.call(data)

    def test_changed_suggestion_keeps_original(self):
        res = self.call(good(i1={"suggested": "この瞬間が好き", "changed": True, "reason_ko": "더 자연스러움",
                                 "tags": ["unnatural_collocation"]}))
        self.assertEqual(res["c01"]["original"], "この一瞬が好き")
        self.assertEqual(res["c01"]["text"], "この一瞬が好き")             # 검수 대상 원문 보존
        self.assertEqual(res["c01"]["suggested"], "この瞬間が好き")


if __name__ == "__main__":
    unittest.main(verbosity=2)
