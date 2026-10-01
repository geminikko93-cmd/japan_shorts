"""일본 팬튜브 쇼츠 자동화 — 웹 UI.

  python -m streamlit run app.py      (또는 run_ui.bat 더블클릭)

① 오늘의 추천 연예인 (인지도·비주얼 기준, 논란 없는 인물 → Claude 선별)
② 칭찬글 · BGM 추천 (스타일·주제 선택 → 기획안 생성/수정, 선택 상태 저장)
③ 영상 편집 (소스 → 장면 분석·자동 배치 → 컷 표/자막 표에서 수정 → AI 검수(선택) → 렌더)
"""
from __future__ import annotations

import base64
import copy
import io
import json
import logging
import os
import shutil
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import streamlit as st

from shorts import director, pipeline, sources
from shorts.common import PipelineError, config_version, load_config, load_dotenv, log, resolve
from shorts.layout import crop_for, get_layout, presets
from shorts.plan import (KINDS, apply_caption_edits, apply_cut_edits, apply_marker_proposals, audio_status,
                         caption_rows, check_plan, clone, cut_rows, is_stale, load_plan, relayout, repetition_warnings,
                         revalidate, save_plan, sources_unchanged)
from shorts.scenes import clip_from_dict
from shorts.styles import DEFAULT_STYLE, LEGACY_STYLE, STYLES, get_style, label as style_label, style_names
from shorts.textrender import subtitle_seconds

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
st.set_page_config(page_title="팬튜브 쇼츠 스튜디오", page_icon="🎬", layout="wide")
ss = st.session_state


def load_cfg() -> None:
    """설정을 읽어 세션에 고정. 파일을 고쳐도 '설정 다시 불러오기'를 누르기 전에는 바뀌지 않는다.

    .env는 다시 읽어 같은 이름의 값을 덮어쓴다. .env에서 지운 키는 이미 실행 중인 프로세스 환경에서
    지워지지 않으므로, 키를 '삭제'했다면 앱을 다시 실행해야 한다.
    """
    ss.cfg = load_config(CONFIG)
    ss.cfg_version = config_version(CONFIG)
    ss.cfg_loaded = time.strftime("%H:%M:%S")


if "cfg" not in ss:
    try:
        load_cfg()
    except PipelineError as e:
        st.error(str(e))
        st.stop()
cfg = ss.cfg
for k, v in {"person": "", "script": None, "cands": [], "result": None, "plan": None, "analysis": None,
             "job": None, "alt_plan": None, "prev_plan": None}.items():
    ss.setdefault(k, v)


def get_layout_names() -> list[str]:
    return list(presets(cfg))


class _UILog(logging.Handler):
    """파이프라인 로그를 st.status 안에 실시간으로 표시."""

    def __init__(self, box):
        super().__init__(logging.INFO)
        self.box, self.lines = box, []

    def emit(self, record):
        self.lines.append(self.format(record))
        self.box.code("\n".join(self.lines[-15:]), language=None)


def run_step(label: str, fn, *args, **kwargs):
    """긴 작업을 진행 상태 박스로 감싸 실행. PipelineError는 화면에 표시하고 None 반환."""
    with st.status(label, expanded=True) as status:
        h = _UILog(st.empty())
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        try:
            out = fn(*args, **kwargs)
        except PipelineError as e:
            status.update(label=f"{label} — 실패", state="error")
            st.error(str(e))
            return None
        finally:
            log.removeHandler(h)
        status.update(label=f"{label} — 완료", state="complete", expanded=False)
        return out


def yt_search_url(q: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote(q)


def recent_scripts() -> list[Path]:
    out = resolve(cfg, cfg["output_dir"])
    return sorted(out.glob("*/script.json"), key=lambda p: -p.stat().st_mtime)[:20] if out.exists() else []


def reset_person() -> None:
    """새 인물로 바꿀 때 이전 인물의 선택·계획·결과를 명시적으로 비움."""
    ss.cands, ss.result, ss.plan, ss.analysis, ss.job, ss.alt_plan, ss.prev_plan = [], None, None, None, None, None, None
    ss.captions = []
    for k in ("cap_editor", "title_pick", "cut_editor", "capev_editor", "style_pick"):
        ss.pop(k, None)


def set_script(script: dict) -> None:
    reset_person()
    ss.script = script
    ss.person = script.get("name_ko") or script.get("name_ja") or ss.person


# ───────────────────────── 사이드바 ─────────────────────────
with st.sidebar:
    st.header("🎬 쇼츠 스튜디오")
    st.caption("상태")
    api = director.api_settings(cfg)
    st.write(("✅ " if api["configured"] else "❌ ") + "Claude API 키 설정 — 추천·칭찬글·검수")
    st.caption(f"모델: {api['model']}")
    st.caption("키 설정 여부만 표시합니다. 실제 연결은 생성 요청 시 확인됩니다.")
    st.write(("✅ " if os.environ.get("YOUTUBE_API_KEY") else "❌ ") + "YouTube API 키 — CC 검색")
    st.write(("✅ " if shutil.which("ffmpeg") else "❌ ") + "`ffmpeg` — 영상 편집")
    st.caption(f"적용된 설정: `{ss.cfg_version}` ({ss.cfg_loaded} 불러옴)")
    if config_version(CONFIG) != ss.cfg_version:
        st.warning("config.yaml이 바뀌었습니다. 아래 버튼을 눌러야 적용됩니다.")
    if st.button("🔁 설정 다시 불러오기", width="stretch",
                 help="config.yaml과 .env를 다시 읽습니다. 이미 만든 편집 계획은 설정이 바뀌면 렌더 전에 '오래된 계획'으로 거부됩니다."):
        try:
            load_cfg()
            load_dotenv(ROOT / ".env")
            st.rerun()
        except PipelineError as e:
            st.error(str(e))

    st.divider()
    st.caption("최근 기획안 불러오기")
    recent = recent_scripts()
    if recent:
        pick = st.selectbox("기획안", recent, format_func=lambda p: p.parent.name, label_visibility="collapsed")
        if st.button("불러오기", width="stretch"):
            warns: list[str] = []
            try:
                set_script(director.load_script(pick, warns))
                for w in warns:
                    st.toast(w)
                st.rerun()
            except PipelineError as e:
                st.error(str(e))
    else:
        st.write("아직 없음")

    if ss.script:
        st.divider()
        st.success(f"작업 중: **{ss.script.get('name_ko', '')}** ({ss.script.get('name_ja', '')})")
    if ss.job:
        j = ss.job
        icon = {"running": "⏳", "failed": "⛔", "done": "✅"}.get(j["state"], "")
        st.caption(f"{icon} 작업 `{j['id']}` — {j['state']}")

tab1, tab2, tab3 = st.tabs(["① 오늘의 추천 연예인", "② 칭찬글 · BGM 추천", "③ 영상 편집"])


# ───────────────────────── ① 오늘의 추천 ─────────────────────────
def recent_names(days: int = 7) -> list[str]:
    """이미 기획안을 만든 인물 + 최근 며칠간 추천받은 인물 (같은 얼굴 반복 방지)."""
    out = resolve(cfg, cfg["output_dir"])
    names = {json.loads(p.read_text(encoding="utf-8")).get("name_ja", "") for p in recent_scripts()}
    cutoff = time.strftime("%Y%m%d", time.localtime(time.time() - days * 86400))
    for f in out.glob("recommend_*.json") if out.exists() else []:
        if f.stem.split("_")[-1] >= cutoff:
            names |= {p["name_ja"] for p in json.loads(f.read_text(encoding="utf-8")).get("people", [])}
    return sorted(n for n in names if n)


with tab1:
    today = time.strftime("%Y%m%d")
    rec_file = resolve(cfg, cfg["output_dir"]) / f"recommend_{today}.json"
    st.subheader(f"오늘의 추천 연예인 · {time.strftime('%Y-%m-%d')}")
    st.caption("일본에서 인지도가 높고 비주얼로 사랑받는 배우·아이돌 중 논란·스캔들이 없는 인물을 추천합니다. "
               "(Claude 1회 호출 · YouTube 할당량 사용 안 함 · 조회수 근거가 아니라 AI 지식 기반)")

    c1, c2, c3 = st.columns([3, 1, 1])
    kinds = c1.multiselect("분야", list(director.RECOMMEND_KINDS), default=list(director.RECOMMEND_KINDS))
    n_people = c2.number_input("추천 인원", 1, 10, 5)
    c3.write("")
    skip_recent = c3.checkbox("최근 인물 제외", True, help="이미 만든 인물과 최근 7일간 추천된 인물은 빼고 새로 추천")
    refresh = st.button("✨ 오늘의 추천 받기" if not rec_file.exists() else "🔄 다른 인물로 다시 추천", type="primary",
                        disabled=not kinds)

    if refresh:
        def _recommend():
            people = director.recommend_people(cfg, int(n_people), kinds, recent_names() if skip_recent else [])
            rec_file.parent.mkdir(parents=True, exist_ok=True)
            rec_file.write_text(json.dumps({"people": people}, ensure_ascii=False, indent=2), encoding="utf-8")
        run_step("인물 추천 (Claude)", _recommend)

    if rec_file.exists():
        rec = json.loads(rec_file.read_text(encoding="utf-8"))
        if not rec["people"]:
            st.warning("추천 결과가 없습니다. 분야를 넓히거나 '최근 인물 제외'를 끄고 다시 시도하세요.")
        st.caption("⚠️ AI 지식 기준이라 아주 최근 논란은 반영되지 않을 수 있습니다. 만들기 전에 '최근 뉴스'로 한 번 확인하세요.")
        cols = st.columns(min(len(rec["people"]), 3) or 1)
        for i, p in enumerate(rec["people"]):
            with cols[i % len(cols)].container(border=True):
                st.markdown(f"### {p['name_ko']}")
                st.caption(f"{p['name_ja']} · {p['kind']}")
                st.write(p["reason_ko"])
                if p.get("known_for_ko"):
                    st.caption("대표작: " + " · ".join(p["known_for_ko"]))
                cc = p.get("cc")
                if cc:
                    st.markdown(f"**CC 영상 {cc['total']}개 · 쓸 만해 보이는 것 {cc['likely']}개**"
                                + (" ✅" if cc["likely"] >= 3 else " ⚠️ 소스 부족" if cc["likely"] == 0 else ""))
                    st.caption(f"{cc['checked']} 확인 · 검색어 {', '.join(cc['keywords'])} · 제목·설명·길이만 본 판단 "
                               "(받아서 분석하면 자막·정지 화면으로 더 빠질 수 있음)")
                    for t_ in cc.get("top", [])[:3]:
                        st.markdown(f"- [{t_['title'][:40]}]({t_['url']}) · {t_['seconds'] // 60}:{t_['seconds'] % 60:02d}")
                if st.button("🔍 CC 영상 확인" + (" (다시)" if cc else "") + " · 약 300 units", key=f"cc_{i}",
                             width="stretch", help="검색어 3개로 CC 영상이 얼마나 있는지 확인합니다 (다운로드 없음). "
                                                   "같은 검색은 24시간 동안 저장된 결과를 다시 써서 할당량을 쓰지 않습니다."):
                    def _check(p=p):
                        r = sources.check_person({"name_ja": p["name_ja"], "name_ko": p["name_ko"]}, cfg)
                        p["cc"] = r
                        rec_file.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                        return r
                    if run_step(f"{p['name_ko']} CC 영상 확인", _check):
                        st.rerun()
                a, b = st.columns(2)
                a.link_button("📰 최근 뉴스", "https://www.google.com/search?tbm=nws&q="
                              + urllib.parse.quote(p["name_ja"]), width="stretch")
                if b.button("만들기 →", key=f"pick_{i}", type="primary", width="stretch"):
                    reset_person()
                    ss.person, ss.script = p["name_ja"], None
                    st.success(f"'{p['name_ko']}' 선택됨 → **② 칭찬글 · BGM 추천** 탭으로 이동하세요.")

# ───────────────────────── ② 칭찬글 · BGM ─────────────────────────
with tab2:
    c1, c2, c3, c4, c5 = st.columns([1.8, 1.3, 1.3, 2.2, 0.9])
    person = c1.text_input("인물 이름 (한국어로 입력해도 됨)", ss.person)
    gen_style = c2.selectbox("영상 스타일", style_names(), index=style_names().index(DEFAULT_STYLE),
                             format_func=style_label,
                             help="새 기획안의 말투·자막 수·편집 호흡. 기본은 청춘·설렘. 효과음은 어떤 스타일에서도 넣지 않습니다.")
    theme = c3.selectbox("이번 영상 주제", list(director.THEMES),
                         help="제목과 칭찬글을 한 주제로 통일합니다. '미소'를 고르면 웃는 장면에 어울리는 문구만 만듭니다.")
    extra = c4.text_input("추가 요청 (선택, 한국어로)", placeholder="원하는 분위기나 강조할 점")
    c5.write("")
    c5.write("")
    if c5.button("✨ 생성", type="primary", width="stretch", disabled=not person.strip()):
        script = run_step(f"'{person}' 칭찬글·BGM 생성 (Claude)", director.generate_script,
                          person.strip(), cfg, extra, theme, gen_style)
        if script:
            job, _ = pipeline.job_dirs(cfg, script, person)
            director.save_script(script, job / "script.json")
            set_script(script)
            st.rerun()
    st.caption(f"{style_label(gen_style)}: {get_style(cfg, gen_style)['description']}")

    script = ss.script
    if not script:
        st.info("① 탭에서 인물을 고르거나 이름을 입력하고 **생성**을 누르세요. 이전 작업은 사이드바에서 불러올 수 있습니다.")
    else:
        st.markdown(f"## {script['name_ko']} <small>({script['name_ja']})</small>", unsafe_allow_html=True)
        sname = script.get("style") or ""
        st.caption(f"스타일: **{style_label(sname) if sname else '정보 없음 (기존 기획안 → 칭찬 나열형으로 취급, 불러오기만으로 바꾸지 않음)'}**"
                   f" · 주제: **{script.get('theme') or '지정 안 함 (주제 혼합)'}**")
        st.write(script.get("profile_ko", ""))
        if script.get("works"):
            st.caption("대표작: " + " · ".join(
                f"{w.get('title_ko') or w['title']} ({w['year']})" + (" ⚠️" if w["confidence"] == "low" else "")
                for w in script["works"]))

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader(f"💬 칭찬글 ({len(script['captions'])}개)")
            sug = get_style(cfg, sname or LEGACY_STYLE)["captions"]["suggested_count"]
            st.caption(f"체크한 문구가 위에서부터 순서대로 쓰입니다 ({style_label(sname or LEGACY_STYLE)} 기준 30초에 "
                       f"약 {sug[0]}~{sug[1]}개로 시작, 강제 아님). 장면 내용을 보고 맞추는 것이 아니므로 ③ 탭에서 확인·교체하세요. "
                       "문구를 고치면 한국어 뜻은 '예전 문구 기준'으로 표시됩니다.")
            used = set(script.get("selection", {}).get("used", []))
            df = pd.DataFrame([{"id": c["id"], "사용": c["id"] in used, "ja": c["ja"], "ko": c["ko"],
                                "뜻 상태": "" if not c["ko"] or c.get("ko_for", c["ja"]) == c["ja"] else "⚠️ 예전 문구 기준"}
                               for c in script["captions"]])
            edited = st.data_editor(
                df, key="cap_editor", width="stretch", hide_index=True, num_rows="dynamic",
                height=min(38 + 35 * len(df), 560), disabled=["id", "뜻 상태"],
                column_config={"id": st.column_config.TextColumn("ID", width="small"),
                               "사용": st.column_config.CheckboxColumn(width="small"),
                               "ja": st.column_config.TextColumn("자막 (일본어)", width="large"),
                               "ko": st.column_config.TextColumn("뜻 (한국어)", width="large")})
            orig = {c["id"]: c for c in script["captions"]}
            sel = []
            for r in edited.to_dict("records"):
                ja = director.clean_text(r.get("ja"))
                if r.get("사용") is True and ja:
                    cid = director.clean_text(r.get("id"))
                    o = orig.get(cid, {})
                    ko = director.clean_text(r.get("ko"))
                    sel.append({"id": cid or f"new{len(sel)}", "ja": ja, "ko": ko,
                                "ko_for": o.get("ko_for", o.get("ja", "")) if ko == o.get("ko") else ja,
                                "kind": o.get("kind", "fan")})
            ss.captions = sel
            target = cfg["video"]["target_duration"]
            secs = sum(subtitle_seconds(c["ja"], cfg) for c in sel)
            if (sname or LEGACY_STYLE) == LEGACY_STYLE:
                st.progress(min(secs / target, 1.0), text=f"선택 {len(sel)}개 · 예상 길이 약 {min(secs, target):.1f}초 / "
                                                          f"목표 {target:.0f}초 (자막 1개 = 컷 1개)")
            else:
                st.caption(f"선택 {len(sel)}개 · 청춘·설렘은 컷 길이를 장면으로 정하고, 코멘트 하나를 여러 컷에 걸쳐 둡니다.")
            for w in repetition_warnings([c["ja"] for c in sel]):
                st.caption("⚠️ " + w)

        with right:
            st.subheader("🎵 BGM 추천")
            st.caption("출연작 주제가/OST 또는 본인 곡 위주. ⚠️ 는 AI가 확신하지 못한 항목이니 직접 확인하세요.")
            for s in script["songs"]:
                with st.container(border=True):
                    a, b = st.columns([4, 1])
                    ko_title = s.get("title_ko") or ""
                    a.markdown(f"**{s['title']}**" + (f" ({ko_title})" if ko_title and ko_title != s["title"] else "")
                               + f" — {s.get('artist_ko') or s['artist']}" + ("  ⚠️" if s["confidence"] == "low" else ""))
                    a.caption(s.get("relation_ko") or s["relation"])
                    b.link_button("▶ 듣기", yt_search_url(f"{s['title']} {s['artist']}"), width="stretch")
            st.info("영상 파일에는 음악을 넣지 않습니다(기본 무음). 업로드할 때 YouTube 앱 **'사운드 추가'**에서 "
                    "이 곡을 검색해 넣는 것이 안전합니다. 곡 제목만으로 후렴 시작 시각을 정하지 않습니다.", icon="ℹ️")

            st.subheader("🏷️ 제목")
            ti = script.get("selection", {}).get("title_idx", 0)
            ss.title_idx = st.radio(
                "제목", range(len(script["titles"])), key="title_pick", label_visibility="collapsed",
                index=ti if ti < len(script["titles"]) else 0,
                format_func=lambda i: f"{script['titles'][i]['ko']}  —  "
                                      f"{script['titles'][i]['line1']} / {script['titles'][i]['line2']}")
            st.caption("해시태그: " + " ".join(script.get("hashtags", [])))

        b1, b2 = st.columns(2)
        if b1.button("💾 칭찬글·선택 상태 저장", width="stretch",
                     help="문구, 사용 체크, 순서, 제목 선택을 ID 기준으로 저장합니다. 다시 불러오면 그대로 복원됩니다."):
            caps, used_ids = [], []
            for r in edited.to_dict("records"):
                ja = director.clean_text(r.get("ja"))
                if not ja:
                    continue
                cid = director.clean_text(r.get("id"))
                o = orig.get(cid, {})
                ko = director.clean_text(r.get("ko"))
                caps.append({"id": cid, "ja": ja, "ko": ko, "kind": o.get("kind", "fan"),
                             "ko_for": (o.get("ko_for") or o.get("ja", "")) if ko == o.get("ko") else (ja if ko else "")})
            script["captions"] = caps
            try:
                fixed = director.normalize_script(script)         # 새 행에 ID 부여 + 구조 검증
            except PipelineError as e:
                st.error(str(e))
            else:
                ids_in_order = [c["id"] for c in fixed["captions"]]
                checked = [r.get("사용") is True for r in edited.to_dict("records") if director.clean_text(r.get("ja"))]
                fixed["selection"] = {"used": [i for i, u in zip(ids_in_order, checked) if u], "title_idx": ss.title_idx}
                job, _ = pipeline.job_dirs(cfg, fixed)
                director.save_script(fixed, job / "script.json")
                ss.script = fixed
                ss.pop("cap_editor", None)
                st.toast(f"저장됨: {job / 'script.json'}")
                st.rerun()
        stale = [c for c in script["captions"] if c["ko"] and c.get("ko_for", c["ja"]) != c["ja"]]
        if b2.button(f"🔤 바뀐 문구만 한국어 뜻 갱신 ({len(stale)}개)", width="stretch", disabled=not stale):
            res = run_step("뜻 갱신 (Claude)", director.translate_ko, [c["ja"] for c in stale], cfg)
            if res:
                for c, ko in zip(stale, res):
                    c["ko"], c["ko_for"] = ko, c["ja"]
                job, _ = pipeline.job_dirs(cfg, script)
                director.save_script(script, job / "script.json")
                ss.pop("cap_editor", None)
                st.rerun()


# ───────────────────────── ③ 영상 편집 ─────────────────────────
LAYOUT_LABELS = {"tall": "세로형 4:5 (빈 공간 적음)", "square": "정사각 1:1 (기존)"}
ROLE_LABELS = {"normal": "일반", "key": "핵심 표정"}
KIND_LABELS = KINDS
FIT_LABELS = {"ok": "장면 일치(사용자 설명 기준)", "mismatch": "장면과 안 맞을 수 있음", "unverified": "장면 적합성 미확인"}


def thumb_uri(clip, layout_name: str, width: int = 150) -> str | None:
    """장면 대표 프레임을 실제 출력과 같은 비율·위치로 크롭한 미리보기 (data URI)."""
    if not clip.thumb or not Path(clip.thumb).exists():
        return None
    from PIL import Image
    lay = get_layout(cfg, layout_name)
    im = Image.open(clip.thumb)
    k = im.width / clip.frame_size[0]
    x, y, w, h = crop_for(clip.face, *clip.frame_size, lay, cfg)
    im = im.crop((round(x * k), round(y * k), round((x + w) * k), round((y + h) * k)))
    im = im.resize((width, round(width * h / w)))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def plan_inputs() -> tuple:
    """계획을 만든 입력. ② 탭 선택이 바뀌면 달라짐 → '다시 배치' 안내 (자동으로 덮어쓰지 않음)."""
    return tuple((c["id"], c["ja"]) for c in ss.get("captions") or []), ss.get("title_idx", 0)


def person_dir() -> Path:
    return pipeline.job_dirs(cfg, ss.script)[0]


def persist(plan) -> None:
    """편집할 때마다 실행 폴더와 인물 폴더(최신 계획)에 저장 → 재시작 후 '불러오기'로 복원."""
    save_plan(plan, Path(plan.run_dir) / "edit_plan.json")
    save_plan(plan, person_dir() / "edit_plan.json")


def set_plan(plan) -> None:
    ss.plan, ss.alt_plan = plan, None
    ss.plan_inputs = plan_inputs()
    for k in ("cut_editor", "capev_editor"):
        ss.pop(k, None)
    persist(plan)


def set_analysis(run_dir: Path, clips: list, srcs: list, info: dict) -> None:
    ss.analysis = {"run_dir": str(run_dir), "clips": {c.id: c for c in clips}, "order": [c.id for c in clips],
                   "srcs": srcs, "info": info}


def build_plan(style: str, first_clip: str = "", exclude: list[str] | None = None, layout: str | None = None):
    an = ss.analysis
    clips = [an["clips"][i] for i in an["order"]]
    return pipeline.make_plan(cfg, ss.script, clips, an["srcs"], ss.get("captions") or [],
                              ss.get("title_idx", 0), Path(an["run_dir"]), layout=layout, style=style,
                              first_clip=first_clip, exclude_sources=exclude,
                              exclude_text=ss.get("flt_text"), exclude_static=ss.get("flt_static"))


def analyze_and_plan(srcs: list, style: str) -> None:
    run_dir = pipeline.new_run_dir(cfg, ss.script)
    info: dict = {}
    clips = pipeline.analyze(cfg, ss.script, srcs, run_dir, info)
    set_analysis(run_dir, clips, srcs, info)
    set_plan(build_plan(style))


def load_saved_plan(path: Path) -> None:
    plan = load_plan(path)
    srcs = pipeline.plan_sources(plan)
    ana = Path(plan.run_dir) / "analysis.json"
    if not ana.exists():
        raise PipelineError(f"이 계획의 장면 분석 결과가 없습니다: {ana}. 다시 분석하세요.")
    rows, info = pipeline.load_analysis(ana)
    by_path = {s.path: s for s in srcs}
    set_analysis(Path(plan.run_dir), [clip_from_dict(d, by_path) for d in rows], srcs, info)
    ss.plan, ss.alt_plan = plan, None
    ss.plan_inputs = plan_inputs()
    ss.style_next = plan.style                        # 불러온 계획의 스타일 그대로 (다음 실행에서 위젯에 반영)
    for k in ("cut_editor", "capev_editor"):
        ss.pop(k, None)


def set_first_clip(plan, cid: str, clips: dict) -> None:
    """cid를 1번 컷으로. 이미 다른 칸에서 쓰면 1번과 자리 바꿈 (자막은 위치 그대로)."""
    rows = cut_rows(plan)
    hit = next((r for r in rows if r["clip_id"] == cid), None)
    if hit:
        hit["clip_id"], rows[0]["clip_id"] = rows[0]["clip_id"], cid
        hit["start"], rows[0]["start"] = rows[0]["start"], hit["start"]
    else:
        rows[0].update(clip_id=cid, start=0.0)
    apply_cut_edits(plan, rows, clips, cfg)
    persist(plan)
    ss.pop("cut_editor", None)


def review_status(c) -> str:
    rv = c.review or {}
    st_ = rv.get("status")
    if not st_:
        return "미검수"
    if st_ == "failed":
        return "⛔ 검수 실패 — 검수 전 문구"
    if st_ == "suggestion_applied":
        return "AI 권장 문구 적용 (재검수 전)" if rv.get("text") == c.text else "적용 후 수정됨 → 재검수 필요"
    if rv.get("text") != c.text:
        return "검수 후 수정됨 → 재검수 필요"
    s = "AI 검수됨 (사람 검수 아님)"
    if rv.get("changed"):
        s += " · 수정 제안 있음"
    return s + " · " + FIT_LABELS.get(rv.get("scene_fit", "unverified"), "")


def do_review(plan) -> None:
    caps = plan.captions
    items = [{"id": "title", "text": f"{plan.title[0]} / {plan.title[1]}", "prev": "", "next": "", "kind": "fan"}]
    for i, c in enumerate(caps):
        items.append({"id": c.id, "text": c.text, "kind": c.kind, "scene_note": c.scene_note,
                      "prev": caps[i - 1].text if i else "", "next": caps[i + 1].text if i + 1 < len(caps) else ""})
    try:
        res = director.review_texts(items, f"{plan.title[0]} / {plan.title[1]}", plan.style, cfg,
                                    person_dir() / "review_cache.json")
    except PipelineError as e:
        for c in caps:      # 실패해도 검수 전 문구를 '검수 완료'로 표시하지 않음
            c.review = {"status": "failed", "error": str(e)[:300], "text": c.text}
        persist(plan)
        raise
    plan.title_review = res["title"]
    for c in caps:
        c.review = res[c.id]
    persist(plan)
    return True


with tab3:
    script = ss.script
    if not script:
        st.info("먼저 ② 탭에서 칭찬글을 생성하거나 기획안을 불러오세요.")
        st.stop()
    job_root, work = pipeline.job_dirs(cfg, script)
    st.subheader(f"{script['name_ko']} 쇼츠 편집")

    captions = ss.get("captions") or []
    title_idx = ss.get("title_idx", 0)
    t = script["titles"][title_idx]
    st.caption(f"제목: **{t['ko']}** ({t['line1']} / {t['line2']}) · 자막 {len(captions)}개 선택됨 (② 탭에서 변경)")
    music = cfg.get("audio", {}).get("music_path")
    if music:
        st.info(f"🎵 BGM 사용 중: `{music}` (config.yaml audio.music_path). 효과음은 넣지 않습니다.")
    else:
        st.caption("🔇 기본 출력은 무음입니다. 업로드 시 음악을 추가할 수 있습니다. (원본 소리·효과음 없음)")
    if not captions:
        st.warning("② 탭에서 사용할 칭찬글을 먼저 체크하세요.")

    default_style = (ss.plan.style if ss.get("plan") else None) or script.get("style") or LEGACY_STYLE
    if "style_next" in ss:
        ss.style_pick = ss.pop("style_next")
    if "style_pick" not in ss:
        ss.style_pick = default_style
    style_now = st.selectbox("편집 스타일", style_names(), key="style_pick", format_func=style_label,
                             help="기존 기획안·계획은 불러오기만으로 스타일을 바꾸지 않습니다. 바꾸려면 아래에서 미리보기 후 적용하세요.")

    tl_cfg = cfg.get("timeline", {})
    f1, f2 = st.columns(2)
    f1.checkbox("원본에 자막·텔롭이 박힌 장면/영상 자동 제외", key="flt_text",
                value=bool(tl_cfg.get("exclude_text_scenes", True)),
                help=f"장면 {int(float(tl_cfg.get('text_source_ratio', 0.3)) * 100)}% 이상에서 글자가 보이면 그 영상은 통째로 뺍니다. "
                     "글자 모양을 보는 간단한 감지라 틀릴 수 있으니 썸네일의 📝 표시로 확인하세요.")
    f2.checkbox("사진 슬라이드쇼·멈춘 화면 자동 제외", key="flt_static",
                value=bool(tl_cfg.get("exclude_static_scenes", True)),
                help="0.4초 사이 화면이 거의 안 바뀌는 장면(🖼)을 뺍니다. 바꾼 뒤에는 '다시 자동 배치'를 누르세요.")

    # ── 1단계: 소스 → 장면 분석 + 자동 배치 (렌더는 하지 않음)
    st.markdown("#### 1. 소스 고르고 장면 분석")
    STEPS = "다운로드 → 장면 분할 → 얼굴 장면 추출 → 자동 배치 (렌더 전 미리보기)"
    max_v = cfg["sources"]["max_videos"]
    mode = st.radio("소스", ["🤖 자동 (CC 영상 알아서 고르기)", "✋ CC 영상 직접 고르기", "📁 내 영상 폴더"],
                    horizontal=True)
    keywords: list[str] = []
    if not mode.startswith("📁"):
        kw = st.text_input("검색 키워드 (쉼표로 구분 · 이름, 이름+행사어, 한국어 이름을 기본으로 넣었습니다. "
                           "로마자 이름 등을 더해도 됩니다)", ", ".join(sources.expand_keywords(script, cfg=cfg)))
        keywords = [k.strip() for k in kw.split(",") if k.strip()]
        pages = int(cfg["sources"].get("search_pages", 1))
        st.caption(f"예상 할당량: 검색어 {len(keywords)}개 × {pages}페이지 = 약 {sources.estimate_units(len(keywords), pages)} "
                   f"units (하루 10,000 · 같은 검색은 {cfg['sources'].get('search_cache_hours', 24)}시간 동안 저장된 결과를 써서 0). "
                   "CC 영상은 일본 방송보다 한국 언론·행사 채널이 올린 것이 많습니다.")

    if mode.startswith("🤖"):
        st.caption(f"CC 라이선스 영상을 검색해 제목에 이름이 들어간 영상부터 최대 {max_v}개 받고, "
                   "가장 많이 나오는 얼굴이 크게 잡힌 장면을 골라 자동 배치합니다. "
                   "'가장 많이 나온 얼굴 = 검색한 인물'이라는 가정이라 다른 인물이 섞일 수 있으니 미리보기에서 확인하세요.")
        if st.button("🤖 자동으로 찾아서 분석하기", type="primary", disabled=not (captions and keywords)):
            def _auto():
                found = sources.search_cc_videos(keywords, cfg, names=[script.get("name_ja", ""), script.get("name_ko", "")])
                if not found:
                    raise PipelineError("CC 라이선스 영상이 없습니다. 키워드를 바꾸거나 내 영상 폴더를 사용하세요.")
                ranked = sources.rank_for_auto(found, [script.get("name_ja", ""), *keywords[:1]])
                analyze_and_plan(sources.download(ranked, work / "src", cfg), style_now)
            run_step(f"CC 검색 → {STEPS}", _auto)

    elif mode.startswith("✋"):
        if st.button("🔍 CC 영상 찾기", disabled=not keywords):
            def _search():
                found = sources.search_cc_videos(keywords, cfg, names=[script.get("name_ja", ""), script.get("name_ko", "")])
                for c, ko in zip(found, director.translate_ko([c["title"] for c in found], cfg)):
                    c["title_ko"] = ko
                return found
            res = run_step("CC 라이선스 영상 검색 + 제목 번역", _search)
            if res is not None:
                ss.cands = res
                if not res:
                    st.warning("CC 라이선스 영상이 없습니다. 키워드를 바꾸거나 내 영상 폴더를 사용하세요.")
        picked: list[dict] = []
        if ss.cands:
            n_likely = sum((c.get("assess") or {}).get("likely", False) for c in ss.cands)
            st.caption(f"후보 {len(ss.cands)}개 (라이선스 재검증 통과) · 쓸 만해 보이는 것 {n_likely}개 — 제목·설명·길이만 본 "
                       f"사전 판단이며, 받은 뒤 분석에서 자막·정지 화면이 더 걸러집니다. 최대 {max_v}개 체크하세요.")
            only_likely = st.checkbox("쓸 만해 보이는 후보만 보기", value=n_likely > 0)
            shown = [c for c in ss.cands if not only_likely or (c.get("assess") or {}).get("likely")]
            cols = st.columns(4)
            for i, c in enumerate(shown):
                with cols[i % 4].container(border=True):
                    if c.get("thumbnail"):
                        st.image(c["thumbnail"], width="stretch")
                    st.markdown(f"[{c.get('title_ko', c['title'])[:45]}]({c['url']})")
                    mins, secs = divmod(c.get("seconds", 0), 60)
                    a_ = c.get("assess") or {}
                    st.caption(f"{c['channel']} · {mins}:{secs:02d}" + (" · ✅ 쓸 만해 보임" if a_.get("likely") else ""))
                    if a_.get("flags"):
                        st.caption(" · ".join(a_["flags"]))
                    if st.checkbox("사용", key=f"cand_{c['video_id']}"):
                        picked.append(c)
        if st.button(f"🔬 선택한 영상 분석하기 ({len(picked)}개)", type="primary",
                     disabled=not (picked and captions)):
            run_step(STEPS, lambda: analyze_and_plan(sources.download(picked, work / "src", cfg), style_now))

    else:
        local_dir = st.text_input("영상 폴더 경로", placeholder=r"C:\Users\...\my_clips",
                                  help="직접 촬영했거나 사용 권한을 확보한 영상만 넣으세요. "
                                       "출처 표기가 필요하면 폴더에 sources.json을 두세요 (작업 폴더의 sources.json 형식 그대로).")
        if st.button("🔬 분석하기", type="primary", disabled=not (local_dir.strip() and captions)):
            run_step(STEPS.replace("다운로드 → ", ""),
                     lambda: analyze_and_plan(sources.from_local_dir(Path(local_dir.strip().strip('"'))), style_now))

    saved = job_root / "edit_plan.json"
    if saved.exists() and not ss.get("plan"):
        if st.button(f"📂 저장된 편집 계획 불러오기 ({saved.parent.name}/edit_plan.json)"):
            run_step("편집 계획 불러오기", load_saved_plan, saved)
            st.rerun()

    # ── 2단계: 미리보기에서 고치기
    plan = ss.get("plan")
    an = ss.get("analysis")
    if plan and an:
        st.divider()
        st.markdown("#### 2. 미리보기에서 고치기")
        info = an.get("info") or {}
        det = {"yunet": "YuNet 얼굴 검출", "haar": "Haar 얼굴 검출(정확도 낮음)", "none": "얼굴 검출 없음(중앙 크롭)"}
        st.caption(f"분석 모드: {det.get(info.get('detector'), info.get('detector', '알 수 없음'))} · "
                   f"동일 인물 필터 {'사용 (가장 많이 나온 얼굴 기준 — 대상 인물 보장 아님)' if info.get('person_filter') else '생략'}"
                   + (f" · 다른 얼굴 장면 {info['person_filter_removed']}개 제외" if info.get("person_filter_removed") else ""))
        for s_ in info.get("skipped", []):
            st.warning(s_)
        st.caption("자동 배치는 '얼굴이 크고 안정적으로 보이는 장면'과 스타일의 컷 길이만 사용합니다. 웃는 장면이나 문구에 맞는 "
                   "장면을 알아보지 않으므로, 첫 컷·표정 마커·자막 위치는 아래 표에서 직접 정하세요. 고친 내용은 렌더에 그대로 쓰입니다.")
        psrcs = pipeline.plan_sources(plan)
        if is_stale(plan, cfg, psrcs):
            keep_ok = sources_unchanged(plan, psrcs)
            st.error("이 계획은 지금 설정·스타일·소스와 맞지 않는 **오래된 계획**이라 그대로는 렌더할 수 없습니다. "
                     + ("소스 파일은 그대로이므로 편집을 유지한 채 현재 설정으로 다시 검증할 수 있습니다."
                        if keep_ok else "소스 파일이 바뀌었거나 이전 형식 계획이라 편집을 유지할 수 없습니다."))
            s1, s2 = st.columns(2)
            if keep_ok and s1.button("🔁 편집 유지하고 현재 설정으로 다시 검증", width="stretch"):
                try:
                    revalidate(plan, an["clips"], cfg, psrcs)
                    persist(plan)
                    ss.pop("cut_editor", None)
                    st.rerun()
                except PipelineError as e:
                    st.error(str(e))
            if s2.button("🔄 같은 분석 결과로 다시 자동 배치 (편집 초기화)", width="stretch"):
                p_ = run_step("자동 배치", build_plan, plan.style, "", plan.excluded_sources, plan.layout)
                if p_:
                    set_plan(p_)
                    st.rerun()
        if ss.get("plan_inputs") != plan_inputs():
            st.warning("② 탭의 칭찬글/제목 선택이 이 계획을 만든 뒤 바뀌었습니다. 반영하려면 "
                       "**다시 자동 배치**를 누르세요 (지금 편집 내용은 사라집니다).")
        for n in plan.notes:
            st.info(n)

        clips = an["clips"]
        names = {s.path: f"{Path(s.path).stem} · {s.title[:30]}" for s in an["srcs"]}

        # 스타일 바꾸기: 적용 전 미리보기
        if style_now != plan.style:
            st.warning(f"현재 계획은 **{style_label(plan.style)}** 입니다. 선택한 **{style_label(style_now)}**로 바꾸려면 "
                       "먼저 미리 배치해 비교하세요 (현재 계획은 그대로 보관).")
            if st.button(f"👀 {style_label(style_now)}로 미리 배치"):
                ss.alt_plan = run_step("다른 스타일로 배치", build_plan, style_now, "", plan.excluded_sources, plan.layout)
            alt = ss.get("alt_plan")
            if alt and alt.style == style_now:
                a1, a2 = st.columns(2)
                for col, p_, name in ((a1, plan, "현재"), (a2, alt, "미리보기")):
                    col.markdown(f"**{name}: {style_label(p_.style)}** — 컷 {len(p_.cuts)}개, 자막 {len(p_.captions)}개, "
                                 f"{p_.total:.1f}초")
                    col.dataframe(pd.DataFrame([{"컷": i + 1, "장면": c.clip_id, "초": round(c.frames / p_.fps, 2),
                                                 "역할": ROLE_LABELS.get(c.role, c.role)} for i, c in enumerate(p_.cuts)]),
                                  hide_index=True, height=200)
                if st.button("✅ 미리보기 계획으로 바꾸기 (현재 계획은 '되돌리기'용으로 보관)"):
                    ss.prev_plan = clone(plan)
                    set_plan(alt)
                    st.rerun()
        if ss.get("prev_plan") and st.button(f"↩️ 이전 계획({style_label(ss.prev_plan.style)})으로 되돌리기"):
            p_ = ss.prev_plan
            ss.prev_plan = None
            ss.style_next = p_.style
            set_plan(p_)
            st.rerun()

        layouts = get_layout_names()
        lc1, lc2 = st.columns([1, 3])
        lay_name = lc1.radio("화면 배치", layouts, index=layouts.index(plan.layout) if plan.layout in layouts else 0,
                             format_func=lambda n: LAYOUT_LABELS.get(n, n))
        if lay_name != plan.layout:
            relayout(plan, lay_name, clips, cfg, pipeline.plan_sources(plan))
            persist(plan)
            ss.pop("cut_editor", None)
            st.rerun()
        lay = get_layout(cfg, plan.layout)
        lc2.caption(f"스타일 {style_label(plan.style)} · 제목 {lay.title_h}px · 영상 {lay.width}×{lay.video_h} · "
                    f"자막 {lay.caption_h}px (영상 아래 전용 영역). 추가 자막은 얼굴을 가리지 않지만, 원본에 박힌 자막 자체는 "
                    "지워지지 않습니다. 글자가 많은 소스는 아래에서 제외하세요.")
        tl = get_style(cfg, plan.style).get("template") or []
        if tl:
            with st.expander("🧭 구성 가이드 (제안값 — 장면에 맞게 바꾸세요)"):
                t0 = 0
                for s_ in tl:
                    end = min(s_["until"], plan.target)
                    st.markdown(f"- **{t0:.0f}~{end:.0f}초** ({ROLE_LABELS[s_['cut']]} 컷): {s_['note']}")
                    t0 = end
                    if end >= plan.target:
                        break

        with st.expander("🎞️ 소스별 장면 보기 · 자막이 많은 소스 제외 · 첫 컷 지정 · 표정 마커 찾기", expanded=False):
            excl = []
            for spath, label_ in names.items():
                ids = [i for i in an["order"] if clips[i].source.path == spath]
                n_t = sum(clips[i].has_text for i in ids)
                n_s = sum(clips[i].static for i in ids)
                ratio = n_t / len(ids) if ids else 0
                auto_out = (plan.filters.get("exclude_text") and ratio >= plan.filters.get("text_source_ratio", 0.3))
                st.markdown(f"**{label_}** — 장면 {len(ids)}개 · 📝 글자 감지 {n_t}개 · 🖼 정지 화면 {n_s}개"
                            + (" · **자막 많은 영상 → 자동 제외됨**" if auto_out else ""))
                if st.checkbox("이 소스 제외 (원본 자막·글자가 많거나 다른 인물)", key=f"ex_{spath}",
                               value=spath in plan.excluded_sources):
                    excl.append(spath)
                cols = st.columns(8)
                for k, cid in enumerate(ids):
                    with cols[k % 8]:
                        uri = thumb_uri(clips[cid], plan.layout, 120)
                        if uri:
                            st.image(uri, width="stretch")
                        st.caption(f"{cid} · {clips[cid].duration:.1f}초" + (" 📝" if clips[cid].has_text else "")
                                   + (" 🖼" if clips[cid].static else ""))
                        if st.button("첫 컷", key=f"first_{cid}", width="stretch"):
                            set_first_clip(plan, cid, clips)
                            st.rerun()
            if sorted(excl) != sorted(plan.excluded_sources):
                st.caption("제외한 소스는 **다시 자동 배치**를 눌러야 반영됩니다.")
            if st.button("🔄 다시 자동 배치 (제외 소스 반영, 첫 컷 유지)"):
                first = plan.cuts[0].clip_id if plan.cuts else ""
                if first and clips[first].source.path in excl:
                    first = ""
                p_ = run_step("자동 배치", build_plan, plan.style, first, excl, plan.layout)
                if p_:
                    set_plan(p_)
                    st.rerun()
            st.markdown("**장면 영상 보기** — 웃는 순간 등을 찾아 컷 표의 '표정 마커'에 장면 기준 초를 적으세요.")
            vc = st.selectbox("장면", an["order"], format_func=lambda i: f"{i} (후보 {clips[i].start:.1f}~{clips[i].end:.1f}초, "
                                                                          f"장면 {clips[i].scene_start:.1f}~{clips[i].scene_end:.1f}초)")
            if vc:
                st.video(clips[vc].source.path, start_time=int(clips[vc].scene_start), end_time=int(clips[vc].scene_end) + 1)
                st.caption(f"장면 시작 = 원본 {clips[vc].scene_start:.2f}초. 마커와 시작은 '원본 시각 − 장면 시작'으로 적습니다. "
                           "컷은 같은 장면 안에서만 늘릴 수 있습니다.")

        errors, warns = check_plan(plan, cfg, lay)
        per: dict[str, list[str]] = {}
        for m in errors + warns:
            key = m.split(":", 1)[0].split(" ")[0:2]
            per.setdefault(" ".join(key), []).append(("⛔ " if m in errors else "⚠️ ") + m.split(":", 1)[-1].strip())

        # 컷 표
        st.markdown(f"**🎬 컷** — {len(plan.cuts)}개 · 총 {plan.total:.2f}초 (목표 {plan.target:.0f}초, 최대 {plan.max_duration:.0f}초)")
        rows = cut_rows(plan)
        cdf = pd.DataFrame([{
            "미리보기": thumb_uri(clips[r["clip_id"]], plan.layout) if r["clip_id"] in clips else None,
            "순서": r["order"], "장면": r["clip_id"], "역할": ROLE_LABELS.get(r["role"], r["role"]),
            "시작(장면 기준 초)": r["start"], "길이(초)": r["seconds"], "표정 마커(장면 기준 초)": r["marker"],
            "장면 길이": round(clips[r["clip_id"]].scene_duration, 2) if r["clip_id"] in clips else None,
            "확인": " / ".join(per.get(f"컷 #{r['order']}", [])) or "✅"} for r in rows])
        cut_ed = st.data_editor(
            cdf, key="cut_editor", hide_index=True, width="stretch", num_rows="dynamic",
            column_config={
                "미리보기": st.column_config.ImageColumn(width="small"),
                "순서": st.column_config.NumberColumn(min_value=0, step=1, width="small"),
                "장면": st.column_config.SelectboxColumn(options=an["order"], width="small"),
                "역할": st.column_config.SelectboxColumn(options=list(ROLE_LABELS.values()), width="small"),
                "시작(장면 기준 초)": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.2f"),
                "길이(초)": st.column_config.NumberColumn(min_value=0.1, max_value=30.0, step=0.1, format="%.2f"),
                "표정 마커(장면 기준 초)": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.2f"),
                "장면 길이": st.column_config.NumberColumn(format="%.2f"),
                "확인": st.column_config.TextColumn(width="large"),
            }, disabled=["미리보기", "장면 길이", "확인"])
        role_back = {v: k for k, v in ROLE_LABELS.items()}
        k1, k2, k3 = st.columns(3)
        if k1.button("✅ 컷 적용", width="stretch"):
            recs = cut_ed.rename(columns={"순서": "order", "장면": "clip_id", "역할": "role", "시작(장면 기준 초)": "start",
                                          "길이(초)": "seconds", "표정 마커(장면 기준 초)": "marker"}).to_dict("records")
            for r in recs:
                r["role"] = role_back.get(r.get("role"), "normal") if isinstance(r.get("role"), str) else "normal"
            try:
                apply_cut_edits(plan, recs, clips, cfg)
                persist(plan)
                ss.pop("cut_editor", None)
                st.rerun()
            except PipelineError as e:
                st.error(str(e))
        if k2.button("🎯 마커 기준 구간 제안", width="stretch",
                     help="표정 마커가 있는 컷의 시작·길이를 마커 앞뒤 맥락(같은 장면 안)으로 바꿉니다. 자동 표정 인식이 아닙니다."):
            try:
                for m in apply_marker_proposals(plan, clips, cfg) or ["표정 마커가 있는 컷이 없습니다 (먼저 '컷 적용')."]:
                    st.toast(m)
                persist(plan)
                ss.pop("cut_editor", None)
                st.rerun()
            except PipelineError as e:
                st.error(str(e))
        if k3.button("🔄 다시 자동 배치 (편집 내용 초기화)", width="stretch"):
            p_ = run_step("자동 배치", build_plan, plan.style, "", plan.excluded_sources, plan.layout)
            if p_:
                set_plan(p_)
                st.rerun()

        # 자막 표
        st.markdown(f"**💬 자막** — {len(plan.captions)}개. 컷과 따로 움직입니다: '시작 컷~끝 컷' 동안 표시, 어떤 컷에도 "
                    "자막이 없을 수 있습니다. 컷 순서를 바꿔도 자막은 그 위치에 남습니다.")
        st.caption("장면 설명은 **사용자가 직접 확인해 적는 칸**입니다 (영상 분석 결과 아님). 원본 대사는 확인한 발화 근거가 있어야 합니다.")
        crows = caption_rows(plan)
        by_cap = {c.id: c for c in plan.captions}
        capdf = pd.DataFrame([{
            "ID": r["id"], "자막": r["text"], "종류": KIND_LABELS.get(r["kind"], r["kind"]), "시작 컷": r["first"],
            "끝 컷": r["last"], "한국어 뜻": r["ko"],
            "뜻 상태": "⚠️ 예전 문구 기준" if r["ko"] and by_cap[r["id"]].ko_for and by_cap[r["id"]].ko_for != r["text"] else "",
            "장면 설명(직접 확인)": r["scene_note"], "발화 근거": r["source_note"],
            "AI 검수": review_status(by_cap[r["id"]]),
            "검수 권장 문구": (by_cap[r["id"]].review or {}).get("suggested", "") if (by_cap[r["id"]].review or {}).get("changed") else "",
            "확인": " / ".join(per.get(f"자막 {r['id']}", [])) or "✅"} for r in crows])
        cap_ed = st.data_editor(
            capdf, key="capev_editor", hide_index=True, width="stretch", num_rows="dynamic",
            column_config={
                "ID": st.column_config.TextColumn(width="small"),
                "자막": st.column_config.TextColumn(width="medium"),
                "종류": st.column_config.SelectboxColumn(options=list(KIND_LABELS.values()), width="small"),
                "시작 컷": st.column_config.NumberColumn(min_value=1, step=1, width="small"),
                "끝 컷": st.column_config.NumberColumn(min_value=1, step=1, width="small"),
            }, disabled=["ID", "뜻 상태", "AI 검수", "검수 권장 문구", "확인"])
        d1, d2, d3, d4 = st.columns(4)
        if d1.button("✅ 자막 적용", width="stretch"):
            recs = cap_ed.rename(columns={"ID": "id", "자막": "text", "종류": "kind", "시작 컷": "first", "끝 컷": "last",
                                          "한국어 뜻": "ko", "장면 설명(직접 확인)": "scene_note",
                                          "발화 근거": "source_note"}).to_dict("records")
            apply_caption_edits(plan, recs)
            persist(plan)
            ss.pop("capev_editor", None)
            st.rerun()
        if d2.button("🔎 AI 일본어 검수 (제목+자막)", width="stretch",
                     help="생성과 별도의 요청으로 선택한 문구만 검수합니다. 같은 문구·문맥은 캐시를 씁니다. "
                          "같은 계열 AI의 검수이며 일본어 모어 화자 검수가 아닙니다."):
            if run_step("일본어 검수 (Claude, 별도 요청)", do_review, plan):
                ss.pop("capev_editor", None)
                st.rerun()
            # 실패: 오류는 run_step이 표시, 각 자막은 '검수 실패 — 검수 전 문구'로 남음
        with_sug = [c.id for c in plan.captions if (c.review or {}).get("changed") and (c.review or {}).get("text") == c.text]
        pick_sug = d3.multiselect("권장 문구를 적용할 자막", with_sug, label_visibility="collapsed",
                                  placeholder="권장 문구 적용할 ID")
        if d3.button("권장 문구 적용", width="stretch", disabled=not pick_sug):
            for c in plan.captions:
                if c.id in pick_sug:
                    rv = c.review
                    c.text, c.ko, c.ko_for = rv["suggested"], rv["ko_meaning"], rv["suggested"]
                    c.review = {**rv, "status": "suggestion_applied", "text": rv["suggested"]}
            persist(plan)
            ss.pop("capev_editor", None)
            st.rerun()
        stale = [c for c in plan.captions if c.text and (not c.ko or c.ko_for != c.text)]
        if d4.button(f"🔤 뜻 갱신 ({len(stale)}개)", width="stretch", disabled=not stale,
                     help="한국어 뜻이 없거나 예전 문구 기준인 자막만 번역합니다."):
            res = run_step("뜻 갱신 (Claude)", director.translate_ko, [c.text for c in stale], cfg)
            if res:
                for c, ko in zip(stale, res):
                    c.ko, c.ko_for = ko, c.text
                persist(plan)
                ss.pop("capev_editor", None)
                st.rerun()
        if plan.captions and any(c.review for c in plan.captions):
            with st.expander("🔎 검수 상세 (AI 검수 — 사람 검수 아님, 오류 가능성 있음)"):
                tr = plan.title_review or {}
                if tr:
                    st.markdown(f"**제목** {plan.title[0]} / {plan.title[1]} → 뜻: {tr.get('ko_meaning', '')}"
                                + (f" · 주의: {tr['ko_nuance']}" if tr.get("ko_nuance") else "")
                                + (f" · 제안: {tr['suggested']} ({tr.get('reason_ko', '')})" if tr.get("changed") else ""))
                for c in plan.captions:
                    rv = c.review or {}
                    if not rv:
                        continue
                    if rv.get("status") == "failed":
                        st.error(f"{c.id}: 검수 실패 — {rv.get('error', '')}")
                        continue
                    tags = ", ".join(director.REVIEW_TAG_KO.get(t_, t_) for t_ in rv.get("tags", []))
                    st.markdown(f"**{c.id}** {rv.get('original', '')} → 뜻: {rv.get('ko_meaning', '')}"
                                + (f" · 주의: {rv['ko_nuance']}" if rv.get("ko_nuance") else "")
                                + (f" · 제안: **{rv['suggested']}** ({rv.get('reason_ko', '')})" if rv.get("changed") else "")
                                + (f" · 대안: {' / '.join(rv['alternatives'])}" if rv.get("alternatives") else "")
                                + (f" · 태그: {tags}" if tags else "")
                                + f" · {FIT_LABELS.get(rv.get('scene_fit', 'unverified'))}")

        # 음악 정보 (렌더에는 영향 없음 — 나중에 같은 위치에 음악을 붙이기 위한 기록)
        with st.expander(f"🎵 음악 기준 — {audio_status(plan.audio)}"):
            au = plan.audio or {}
            ref = au.get("reference", {})
            m1, m2, m3 = st.columns(3)
            r_title = m1.text_input("기준 곡", ref.get("title", ""), placeholder="예: 君がくれた夏")
            r_artist = m2.text_input("아티스트", ref.get("artist", ""), placeholder="예: 家入レオ")
            r_ver = m3.text_input("버전/파일", ref.get("version", "") or au.get("file", ""), placeholder="예: TV size / 내 파일 경로")
            m4, m5 = st.columns([1, 2])
            off = m4.text_input("음악 시작 오프셋(초)", "" if au.get("offset") is None else str(au["offset"]),
                                help="직접 확인한 값만. 곡 제목만으로 후렴 시작을 추측하지 않습니다.")
            marks = m5.text_input("주요 마커(곡 기준 초, 쉼표)", ", ".join(str(x) for x in au.get("markers", [])))
            confirmed = st.checkbox("위 오프셋·마커를 실제 음원을 들으며 직접 확인했습니다", au.get("timing") == "user_marked")
            if st.button("음악 기준 저장"):
                try:
                    ms = [float(x) for x in marks.replace("，", ",").split(",") if x.strip()]
                    plan.audio = {**au, "reference": {"title": r_title.strip(), "artist": r_artist.strip(),
                                                      "version": r_ver.strip()},
                                  "offset": float(off) if off.strip() else None, "markers": ms,
                                  "timing": "user_marked" if confirmed and (ms or off.strip()) else "unverified"}
                    persist(plan)
                    st.rerun()
                except ValueError:
                    st.error("오프셋·마커는 숫자로 적으세요.")
            st.caption("기본 MP4는 무음입니다. 여기 정보는 편집 계획에만 저장되며 음악을 내려받거나 자동으로 넣지 않습니다.")

        for e in [e for e in errors if not (e.startswith("컷 #") or e.startswith("자막 "))]:
            st.error(e)
        for w in [w for w in warns if not (w.startswith("컷 #") or w.startswith("자막 "))]:
            st.caption("⚠️ " + w)

        # ── 3단계: 렌더 (계획 그대로)
        st.divider()
        st.markdown("#### 3. 렌더링")
        if errors:
            st.error(f"고쳐야 할 항목 {len(errors)}개가 있어 렌더할 수 없습니다 (표의 ⛔ 확인).")
        if st.button("🎬 이 계획대로 렌더링", type="primary", disabled=bool(errors)):
            jid = pipeline.new_id()
            ss.job = {"id": jid, "state": "running"}
            res = run_step(f"렌더 (작업 {jid})", pipeline.render_plan, copy.deepcopy(cfg), script, plan, "", jid)
            if res:
                out, meta = res
                ss.job = {"id": jid, "state": "done"}
                ss.result = {"job_id": jid, "out": str(out), "meta": meta, "dir": str(Path(out).parent)}
            else:
                ss.job = {"id": jid, "state": "failed"}

    res = ss.get("result")
    if res:
        is_prev = bool(ss.job and ss.job["id"] != res["job_id"])
        if is_prev:
            st.warning(f"아래는 **이전 작업 결과** (작업 {res['job_id']})입니다. 방금 실행한 작업 {ss.job['id']}은(는) "
                       f"{'실패' if ss.job['state'] == 'failed' else '진행 중'}입니다.")
        else:
            st.success(f"완성 (작업 {res['job_id']}): {res['out']}")
        v, m = st.columns([1, 2], gap="large")
        with v:
            st.video(res["out"])
            with open(res["out"], "rb") as f:
                st.download_button("⬇️ 영상 다운로드", f, file_name=Path(res["out"]).name, mime="video/mp4",
                                   width="stretch")
        with m:
            meta = res["meta"]
            st.markdown("**업로드용 제목** (일본 시청자용이라 일본어로 올라갑니다)")
            st.code(meta["title"], language=None)
            st.caption(f"뜻: {meta.get('title_ko', '')}")
            st.markdown("**설명란 (출처 표기 포함)**")
            st.code(meta["description"], language=None)
            st.caption(f"작업 폴더: {res['dir']} (영상·기획안·편집 계획·출처·설정 사본·job.json). "
                       "CLI `make --plan <폴더>/edit_plan.json`으로 같은 편집을 다시 렌더할 수 있습니다.")
