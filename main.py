"""일본 팬튜브 쇼츠 자동화 CLI.

  python main.py trends  --query "#shorts" --query "女優"
  python main.py script  --person "이시하라 사토미" --theme 미소
  python main.py make    --person "이시하라 사토미"                    # 기획→CC소스→편집→출력
  python main.py make    --script examples/script.json --sources-dir my_clips   # API 없이
  python main.py make    --script ... --sources-dir my_clips --plan-only      # 분석+자동 배치만 → edit_plan.json
  python main.py make    --plan output/<이름>/edit_plan.json                   # 고친 계획 그대로 렌더
  python main.py batch   --file people.txt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from shorts import director, pipeline, publish, sources, trends
from shorts.common import PipelineError, load_config, log, require_ffmpeg, resolve, setup_logging, slugify
from shorts.plan import audio_status, load_plan, save_plan
from shorts.styles import DEFAULT_STYLE, label as style_label, style_names


def cmd_trends(cfg: dict, a: argparse.Namespace) -> None:
    out = resolve(cfg, cfg["output_dir"]) / f"trends_{time.strftime('%Y%m%d_%H%M')}.csv"
    rows = trends.scan(cfg, a.query, out)
    print("업로드 이후 시간당 평균 조회수(최근 증가량 아님) / 총조회수 / 카테고리 / 제목 — 짧은 영상 후보")
    for r in rows[:10]:
        print(f"{r['avg_views_per_hour_since_upload']:>9,}/h  {r['views']:>11,}  {r['category']:<8} {r['title'][:40]}")


def cmd_script(cfg: dict, a: argparse.Namespace) -> Path:
    script = director.generate_script(a.person, cfg, a.request or "", a.theme or "", a.style)
    path = resolve(cfg, cfg["output_dir"]) / slugify(script["name_ja"] or a.person) / "script.json"
    director.save_script(script, path)
    print(f"\n[{script['name_ja']}] {script['profile_ko']}")
    print("연관 음악 (confidence=low는 직접 확인):")
    for s in script["songs"]:
        print(f"  - {s['title']} / {s['artist']} ({s['relation']}) [{s['confidence']}]")
    print("제목 후보:")
    for i, t in enumerate(script["titles"]):
        print(f"  [{i}] {t['line1']} / {t['line2']}  ({t['ko']})")
    print(f"→ {path}")
    return path


def cmd_make(cfg: dict, a: argparse.Namespace) -> Path | None:
    require_ffmpeg()
    if getattr(a, "plan", None):                       # 사람이 고친 계획을 그대로 렌더 (자동 배치 안 함)
        plan = load_plan(Path(a.plan))
        script = director.load_script(Path(a.script) if a.script else Path(a.plan).with_name("script.json"))
        out, meta = pipeline.render_plan(cfg, script, plan, a.music_credit or "")
        _finish(cfg, a, out, meta, pipeline.job_dirs(cfg, script)[0])
        return out

    script = (director.load_script(Path(a.script)) if a.script
              else director.generate_script(a.person, cfg, "", getattr(a, "theme", None) or "",
                                            getattr(a, "style", None) or DEFAULT_STYLE))
    if not script.get("name_ja"):
        script["name_ja"] = a.person or "untitled"
    job, work = pipeline.job_dirs(cfg, script)

    if a.sources_dir:
        srcs = sources.from_local_dir(Path(a.sources_dir))
    else:
        names = [script.get("name_ja", ""), script.get("name_ko", "")]
        cands = sources.search_cc_videos(sources.expand_keywords(script, cfg=cfg), cfg, names=names)
        srcs = sources.download(sources.rank_for_auto(cands, names), work / "src", cfg)

    captions = script["captions"][a.caption_start:]
    run_dir = pipeline.new_run_dir(cfg, script)
    clips = pipeline.analyze(cfg, script, srcs, run_dir)
    plan = pipeline.make_plan(cfg, script, clips, srcs, captions, a.title_idx, run_dir,
                              layout=getattr(a, "layout", None), style=getattr(a, "style", None),
                              first_clip=getattr(a, "first_clip", None) or "")
    print(f"스타일: {style_label(plan.style)} / 컷 {len(plan.cuts)}개 / 자막 {len(plan.captions)}개 / "
          f"{plan.total:.1f}초 / {audio_status(plan.audio)}")
    for n in plan.notes:
        print(f"※ {n}")
    if getattr(a, "plan_only", False):
        job.mkdir(parents=True, exist_ok=True)
        director.save_script(script, job / "script.json")
        path = save_plan(plan, job / "edit_plan.json")
        print(f"\n장면 썸네일: {run_dir / 'thumbs'}\n편집 계획: {path}\n"
              f"  cuts(clip_id·src_start·frames)와 captions(text·first_cut·last_cut)를 고친 뒤 --plan \"{path}\" 로 렌더하세요.\n"
              f"  첫 컷만 바꾸려면 --first-clip <장면ID> 로 다시 실행하세요.")
        return None
    out, meta = pipeline.render_plan(cfg, script, plan, a.music_credit or "")
    _finish(cfg, a, out, meta, job)
    return out


def _finish(cfg: dict, a: argparse.Namespace, out: Path, meta: dict, job: Path) -> None:
    if a.upload:
        publish.upload(out, meta, resolve(cfg, "secrets/client_secret.json"),
                       resolve(cfg, "secrets/token.json"), privacy="private")
    print(f"\n완성 영상: {out}\n작업 폴더: {out.parent} (metadata.json·edit_plan.json·job.json 포함)")


def cmd_cc_check(cfg: dict, a: argparse.Namespace) -> None:
    """인물별 CC 영상 미리 확인 (다운로드 없음, 검색어 3개 = 약 300 units, 같은 검색은 24시간 캐시)."""
    for name in a.person:
        ja, _, ko = name.partition("/")
        r = sources.check_person({"name_ja": ja.strip(), "name_ko": ko.strip()}, cfg)
        print(f"\n[{ja}] CC 영상 {r['total']}개 · 쓸 만해 보이는 것 {r['likely']}개 "
              f"(API 약 {r['units']} units, 저장된 결과 {r['cached']}회) — 검색어 {', '.join(r['keywords'])}")
        for t in r["top"]:
            print(f"  - {t['title'][:60]} ({t['seconds'] // 60}:{t['seconds'] % 60:02d}) {t['url']}  {' / '.join(t['flags'])}")


def cmd_batch(cfg: dict, a: argparse.Namespace) -> None:
    people = [ln.strip() for ln in Path(a.file).read_text(encoding="utf-8-sig").splitlines()
              if ln.strip() and not ln.startswith("#")]
    ok, failed = [], []
    for p in people:
        ns = argparse.Namespace(person=p, script=None, sources_dir=None, caption_start=0,
                                title_idx=0, music_credit=None, upload=a.upload, theme=a.theme, style=a.style,
                                layout=None, first_clip=None, plan_only=False, plan=None)
        try:
            ok.append((p, cmd_make(cfg, ns)))
        except PipelineError as e:      # 한 명 실패해도 다음 인물 계속
            log.error("[%s] 실패: %s", p, e)
            failed.append(p)
    print(f"\n배치 완료: 성공 {len(ok)} / 실패 {len(failed)} {failed if failed else ''}")


def main() -> int:
    themes = list(director.THEMES)
    ap = argparse.ArgumentParser(description="일본 팬튜브 쇼츠 자동화")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("trends", help="최근 24시간 안에 업로드된 일본 짧은 영상의 업로드 이후 시간당 평균 조회수 → CSV")
    p.add_argument("--query", action="append", default=[])

    p = sub.add_parser("script", help="인물 기획안(제목/자막/연관 음악) 생성")
    p.add_argument("--person", required=True)
    p.add_argument("--request", help="추가 요청 (예: 외모 칭찬 제목 더)")
    p.add_argument("--theme", choices=themes, help="이번 영상 주제 (제목·자막을 이 주제로 통일)")
    p.add_argument("--style", choices=style_names(), default=DEFAULT_STYLE, help="편집 스타일 (기본: youth_romance 청춘·설렘)")

    p = sub.add_parser("make", help="쇼츠 1편 제작")
    p.add_argument("--person")
    p.add_argument("--script", help="기존 script.json 사용 (LLM 호출 생략)")
    p.add_argument("--sources-dir", help="보유/라이선스 확보한 영상 폴더 (검색·다운로드 생략)")
    p.add_argument("--title-idx", type=int, default=0)
    p.add_argument("--caption-start", type=int, default=0, help="같은 인물로 여러 편 만들 때 자막 시작 위치")
    p.add_argument("--music-credit", help="설명란에 넣을 음원 표기")
    p.add_argument("--upload", action="store_true", help="비공개로 YouTube 업로드")
    p.add_argument("--theme", choices=themes, help="--person으로 기획안을 만들 때의 주제")
    p.add_argument("--layout", help="화면 배치 프리셋 (config layout.presets: tall / square)")
    p.add_argument("--style", choices=style_names(),
                   help="편집 스타일. 생략하면 기획안의 스타일 (스타일 정보가 없는 기존 기획안은 praise_list)")
    p.add_argument("--first-clip", help="첫 컷 장면 ID (예: S1-03). --plan-only로 썸네일을 본 뒤 지정")
    p.add_argument("--plan-only", action="store_true", help="분석·자동 배치만 하고 edit_plan.json 저장 (렌더 안 함)")
    p.add_argument("--plan", help="저장된 edit_plan.json을 그대로 렌더 (script.json은 같은 폴더 또는 --script)")

    p = sub.add_parser("cc-check", help="인물별 CC 영상 미리 확인 (다운로드 없음)")
    p.add_argument("--person", action="append", required=True, help='일본어 이름 또는 "일본어/한국어" (여러 번 가능)')

    p = sub.add_parser("batch", help="people.txt의 인물별로 연속 제작")
    p.add_argument("--file", required=True)
    p.add_argument("--upload", action="store_true")
    p.add_argument("--theme", choices=themes)
    p.add_argument("--style", choices=style_names(), default=DEFAULT_STYLE)

    a = ap.parse_args()
    setup_logging(a.verbose)
    try:
        cfg = load_config(a.config)
        if a.cmd == "make" and not (a.person or a.script or a.plan):
            raise PipelineError("--person, --script, --plan 중 하나는 필요합니다.")
        {"trends": cmd_trends, "script": cmd_script, "make": cmd_make, "batch": cmd_batch,
         "cc-check": cmd_cc_check}[a.cmd](cfg, a)
    except PipelineError as e:
        log.error("%s", e)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
