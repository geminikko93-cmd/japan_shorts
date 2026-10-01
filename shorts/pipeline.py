"""③~⑦ 제작 단계. CLI(main.py)와 UI(app.py) 공용.

  analyze()      소스 → 장면 분석 (실행별 폴더에 썸네일, analysis.json — 실제 검출 모드 기록)
  make_plan()    장면 + 자막 + 스타일 → 편집 계획 (edit_plan.json). 첫 컷/핵심 컷 지정·소스 제외 가능
  render_plan()  계획을 그대로 렌더 (자동 배치로 덮어쓰지 않음). 오래된 계획은 거부
                 결과는 output/<인물>/jobs/<job_id>/ 에 영상·기획안·계획·출처·메타데이터·설정 사본을 함께 보관
  render()       위 세 단계를 한 번에 (기존 호출 호환)
"""
from __future__ import annotations

import copy
import json
import platform
import time
import traceback
import uuid
from pathlib import Path

from . import composer, director, publish, scenes, sources
from .common import PipelineError, config_snapshot, log, require_ffmpeg, resolve, slugify
from .layout import get_layout
from .plan import EditPlan, auto_plan, new_plan, save_plan, scene_filters, validate_for_render
from .styles import LEGACY_STYLE


def job_dirs(cfg: dict, script: dict, fallback: str = "untitled") -> tuple[Path, Path]:
    """(output/<이름>, work/<이름>) 경로. 인물별 최신 기획안·계획은 output/<이름>/ 에 둔다."""
    name = slugify(script.get("name_ja") or fallback)
    return resolve(cfg, cfg["output_dir"]) / name, resolve(cfg, cfg["work_dir"]) / name


def new_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def new_run_dir(cfg: dict, script: dict) -> Path:
    """실행마다 따로 쓰는 작업 폴더 work/<이름>/runs/<시각>_<무작위>."""
    _, work = job_dirs(cfg, script)
    d = work / "runs" / new_id()
    d.mkdir(parents=True, exist_ok=False)
    return d


def load_analysis(path: Path) -> tuple[list[dict], dict]:
    """analysis.json → (장면 목록, 분석 정보). 이전 형식(목록만)도 읽음."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {"detector": "unknown", "skipped": ["이전 형식 분석 결과: 검출 모드 기록 없음"]}
    return data["clips"], data.get("info", {})


def analyze(cfg: dict, script: dict, srcs: list[sources.Source], run_dir: Path,
            info: dict | None = None) -> list[scenes.Clip]:
    require_ffmpeg()
    if not srcs:
        raise PipelineError("소스 영상이 없습니다.")
    info = info if info is not None else {}
    clips = scenes.analyze(srcs, cfg, resolve(cfg, "assets/models"), run_dir / "thumbs", info)
    (run_dir / "analysis.json").write_text(
        json.dumps({"version": 2, "info": info, "clips": [c.to_dict() for c in clips]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return clips


def make_plan(cfg: dict, script: dict, clips: list[scenes.Clip], srcs: list[sources.Source], captions: list,
              title_idx: int, run_dir: Path, *, layout: str | None = None, style: str | None = None,
              first_clip: str = "", climax_clip: str = "", exclude_sources: list[str] | None = None,
              exclude_text: bool | None = None, exclude_static: bool | None = None) -> EditPlan:
    """captions: [{id, ja, ko, kind}] 또는 문자열 목록. style 미지정이면 기획안의 스타일(없으면 기존 동작)."""
    if not 0 <= title_idx < len(script["titles"]):
        raise PipelineError(f"제목 번호 범위 초과 (0~{len(script['titles']) - 1})")
    style = style or script.get("style") or LEGACY_STYLE
    lay = get_layout(cfg, layout)
    flt = scene_filters(cfg, exclude_text, exclude_static)
    cuts, caps, notes = auto_plan(clips, captions, cfg, lay, style=style, first_clip=first_clip,
                                  climax_clip=climax_clip, exclude_sources=exclude_sources, filters=flt)
    t = script["titles"][title_idx]
    used = [s for s in srcs if s.path not in set(exclude_sources or [])]
    plan = new_plan(script.get("name_ja", ""), title_idx, (t["line1"], t["line2"]), lay, cfg, style, cuts, caps,
                    notes, used, run_dir, first_clip, exclude_sources)
    plan.filters = flt
    save_plan(plan, run_dir / "edit_plan.json")
    return plan


def plan_sources(plan: EditPlan) -> list[sources.Source]:
    fields = set(sources.Source.__dataclass_fields__)
    return [sources.Source(**{k: v for k, v in s.items() if k in fields}) for s in plan.sources]


def environment() -> dict:
    """검증·재현용 실행 환경 기록."""
    import subprocess
    try:
        ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        ff = "unknown"
    return {"python": platform.python_version(), "platform": platform.platform(), "ffmpeg": ff}


def _write_job(job_dir: Path, info: dict) -> None:
    (job_dir / "job.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def render_plan(cfg: dict, script: dict, plan: EditPlan, music_credit: str = "",
                job_id: str | None = None) -> tuple[Path, dict]:
    """계획대로 렌더 → (영상 경로, 업로드 메타데이터). 계획을 바꾸지 않는다.

    시작 시점의 설정 복사본을 쓴다. 실패하면 job.json에 failed와 오류를 남기고 예외를 그대로 올린다.
    """
    cfg = copy.deepcopy(cfg)
    require_ffmpeg()
    srcs = plan_sources(plan)
    lay = validate_for_render(plan, cfg, srcs)
    person_dir, _ = job_dirs(cfg, script)
    job_id = job_id or new_id()
    job_dir = person_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    info = {"job_id": job_id, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "person": plan.person, "style": plan.style, "layout": plan.layout, "title": list(plan.title),
            "planned_seconds": round(plan.total, 3), "cuts": len(plan.cuts), "captions": len(plan.captions),
            "audio": plan.audio, "notes": plan.notes, "environment": environment()}
    _write_job(job_dir, info)
    try:
        director.save_script(script, job_dir / "script.json")
        sources.save_sources(srcs, job_dir / "sources.json")
        save_plan(plan, job_dir / "edit_plan.json")
        (job_dir / "config_snapshot.json").write_text(
            json.dumps(config_snapshot(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        out = composer.compose(plan, lay, cfg, Path(plan.run_dir) / f"render_{job_id}",
                               job_dir / f"{person_dir.name}_{job_id}.mp4")
        meta = publish.build_metadata(script, plan.title_idx, srcs, music_credit)
        meta["title"] = f"{plan.title[0]}{plan.title[1]}"[:90] + " #shorts"   # 계획에 저장된 제목 그대로
        publish.save_metadata(meta, job_dir)
    except Exception as e:
        info.update(status="failed", finished=time.strftime("%Y-%m-%d %H:%M:%S"), error=str(e)[-2000:],
                    traceback=traceback.format_exc()[-4000:])
        _write_job(job_dir, info)
        raise
    info.update(status="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"), video=out.name)
    _write_job(job_dir, info)
    # 인물 폴더에는 최신 기획안·계획 사본 (불러오기 편의). 과거 영상의 기록은 jobs/<job_id>/에 남는다
    director.save_script(script, person_dir / "script.json")
    save_plan(plan, person_dir / "edit_plan.json")
    for n in plan.notes:
        log.warning("%s", n)
    return out, meta


def render(cfg: dict, script: dict, srcs: list[sources.Source], captions: list,
           title_idx: int = 0, music_credit: str = "", style: str | None = None) -> tuple[Path, dict]:
    """확보한 소스로 쇼츠 1편을 자동 배치로 렌더 (기존 CLI 호환)."""
    require_ffmpeg()
    if not captions:
        raise PipelineError("사용할 자막이 없습니다.")
    run_dir = new_run_dir(cfg, script)
    clips = analyze(cfg, script, srcs, run_dir)
    plan = make_plan(cfg, script, clips, srcs, captions, title_idx, run_dir, style=style)
    return render_plan(cfg, script, plan, music_credit)
