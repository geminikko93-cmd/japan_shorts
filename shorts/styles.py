"""편집 스타일 프리셋 (데이터). 코드 곳곳에 분기를 두지 않고 여기 값만 읽는다.

- youth_romance(청춘·설렘): 신규 프로젝트 기본. 표정 중심, 컷을 길게, 짧은 팬 코멘트를 여러 컷에 걸쳐 유지
- praise_list(칭찬 나열형): 기존 동작. 자막 1개 = 컷 1개, 노란 대형 제목

값은 모두 '테스트할 편집 가설'의 출발값이며 일본 시청자 취향을 보장하지 않는다.
어떤 스타일에도 효과음 항목은 없다 (composer에 효과음 경로 자체가 없음).
config.yaml의 styles.<이름> 으로 일부 값을 덮어쓸 수 있다.
"""
from __future__ import annotations

import copy

from .common import PipelineError

DEFAULT_STYLE = "youth_romance"   # 새로 만드는 기획안/계획
LEGACY_STYLE = "praise_list"      # 스타일 정보가 없는 기존 기획안/계획 (불러오기만으로 바꾸지 않음)

STYLES: dict[str, dict] = {
    "youth_romance": {
        "label": "청춘·설렘",
        "description": "표정 변화와 자연스러운 미소 중심. 컷을 길게 두고 짧고 담백한 팬 코멘트를 여러 컷에 걸쳐 유지.",
        "cut": {
            "normal_sec": 2.5,        # 일반 컷 (2~3초 출발값)
            "key_sec": 3.5,           # 핵심 표정 컷 (3~4초 출발값, 소스가 짧으면 늘리지 않음)
            "min_sec": 1.5,           # 이보다 짧게 쓸 수 있는 장면만 남으면 컷으로 쓰지 않음
            "marker_before_sec": 1.0,  # 표정 마커 앞 맥락 (웃기 직전 눈빛)
            "marker_after_sec": 2.5,   # 마커 뒤 (미소가 번지는 순간 + 여운)
        },
        "captions": {
            "mode": "hold",           # 코멘트 하나를 여러 컷에 걸쳐 유지
            "suggested_count": [4, 6],  # 30초 기준 초기 후보 수 (강제 아님)
            "line_chars": [8, 14],    # 한 줄 글자 수 안내 (실제 판단은 표시 폭)
            "leave_key_cuts_empty": False,  # True면 핵심 표정 컷은 자막 없이 (선택 사항, 기본 꺼짐)
        },
        # 30초 구성 템플릿 (끝 시각 기준). 실제 장면에 따라 사람이 바꾼다.
        "template": [
            {"until": 3, "cut": "key", "note": "제목이 약속한 표정으로 바로 시작 (로고 인트로 없음)"},
            {"until": 10, "cut": "normal", "note": "첫 장면 분위기를 잇는 자연스러운 행동"},
            {"until": 22, "cut": "normal", "note": "한 가지 주제의 다른 순간들, 자막은 필요한 곳에만"},
            {"until": 28, "cut": "key", "note": "가장 기억에 남는 표정·행동을 충분히"},
            {"until": 999, "cut": "normal", "note": "자연스러운 끝맺음 (억지 루프 없음)"},
        ],
        "title": {"size": 96, "color": "#FFFFFF", "accent_color": "#FFD9E4", "stroke_color": "#000000",
                  "stroke_width": 8, "min_size_ratio": 0.75},
        "subtitle": {"size": 84, "color": "#FFFFFF", "stroke_color": "#000000", "stroke_width": 7,
                     "box_color": [0, 0, 0, 190], "min_size_ratio": 0.8},
        "prompt_ja": (
            "チャンネルの口調: 成人の俳優を好きなファンの、短く落ち着いた口語。広告コピー、流行語の羅列、"
            "キャラクターの口調、敬語とタメ口の混在は避ける。自然な口語の省略はそのままでよい。\n"
            "「〜すぎる」「国宝級」「尊い」「沼」「優勝」などの定番表現は使ってもよいが、1本の中で同じ語尾・同じ意味の褒め言葉を繰り返さない。\n"
            "表情やしぐさの一瞬に反応する短い一言を優先する(参考例: この笑顔が好き / 笑った瞬間、かわいすぎる / 何回でも見たくなる。"
            "例をそのまま多用しない)。映像から性格・演技力・事実を断定しない。本人の発言のような引用(「」)を作らない。\n"
            "タイトルは『名前 + 感情のポイント1つ』。テロップは1行8〜14字程度を目安にする(表示幅が優先)。"),
    },
    "praise_list": {
        "label": "칭찬 나열형 (기존)",
        "description": "기존 동작. 자막 1개 = 컷 1개, 1.5~2초마다 칭찬 한 줄, 노란 대형 제목.",
        "cut": {"min_sec": 1.5},
        "captions": {"mode": "per_cut", "suggested_count": [12, 18], "line_chars": [8, 14],
                     "leave_key_cuts_empty": False},
        "template": [],
        "title": {},                  # config.yaml title 그대로 (노란 대형)
        "subtitle": {},
        "prompt_ja": ("テロップは日本のネット民が実際に書くような自然な口語(例:「〜すぎる」「国宝級」「この顔は一生見てられる」)。"
                      "ただし同じ語尾を何度も繰り返さない。"),
    },
}

TEXT_KEYS = {"size", "color", "accent_color", "stroke_color", "stroke_width", "box_color", "min_size_ratio"}


def style_names() -> list[str]:
    return list(STYLES)


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def get_style(cfg: dict, name: str | None) -> dict:
    name = name or DEFAULT_STYLE
    if name not in STYLES:
        raise PipelineError(f"스타일 '{name}'이(가) 없습니다. 사용 가능: {', '.join(STYLES)}")
    st = _merge(STYLES[name], (cfg.get("styles") or {}).get(name, {}))
    st["name"] = name
    st.pop("sfx", None)          # 혹시 사용자 설정에 효과음 키가 있어도 무시
    return st


def styled_cfg(cfg: dict, name: str | None) -> dict:
    """스타일의 글자 모양을 title/subtitle에 덮어쓴 설정 복사본 (글꼴은 사용자 설정 유지)."""
    st = get_style(cfg, name)
    out = copy.deepcopy(cfg)
    for part in ("title", "subtitle"):
        out[part] = {**out[part], **{k: v for k, v in st.get(part, {}).items() if k in TEXT_KEYS}}
    out["_style"] = st["name"]
    return out


def label(name: str) -> str:
    return STYLES.get(name, {}).get("label", name)
