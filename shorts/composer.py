"""⑥ 합성: 계획(edit_plan)대로 컷 나열 + 얼굴 크롭 + 제목 + 자막 타임라인 → 1080x1920 MP4.

영상 컷과 자막은 서로 독립: 자막은 여러 컷에 걸치거나, 어떤 컷에는 없을 수 있다.

캡컷에서 하던 'W로 자르고 Q/E로 날리기 → 영상 나열 → 확대 → 텍스트'를 ffmpeg로 처리합니다.
오디오: 원본 소리는 항상 제거. 기본은 무음 AAC 트랙(영상 길이와 같음).
config audio.music_path를 직접 지정한 경우에만 그 음악을 넣고, 음악이 짧으면 나머지는 무음.
효과음은 어떤 스타일에서도 넣지 않습니다 (예전 config의 sfx_dir/cut_sfx/sfx_volume 키는 무시).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .common import PipelineError, log, resolve, run_ffmpeg
from .layout import Layout
from .plan import EditPlan, VideoClipEvent
from .styles import styled_cfg
from .textrender import render_subtitle, render_title

AUDIO_RATE = 44100


def _render_segment(slot: VideoClipEvent, out: Path, lay: Layout, fps: int) -> None:
    """slot.src_start부터 정확히 slot.frames 프레임.

    fps=...:round=down → 출력 프레임 k는 시각 (k+1)/fps 이전의 원본 프레임만 사용하므로
    src_start + frames/fps (= 장면 끝 이하) 이후의 프레임은 나오지 않는다. 끝 패딩(tpad)도 쓰지 않는다.
    """
    x, y, w, h = slot.crop
    vf = (f"crop={w}:{h}:{x}:{y},scale={lay.width}:{lay.video_h}:flags=lanczos,setsar=1,"
          f"fps={fps}:round=down")
    run_ffmpeg(["-ss", f"{slot.src_start:.4f}", "-i", slot.source, "-vf", vf,
                "-frames:v", str(slot.frames), "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                str(out)], f"세그먼트 {out.name}")
    got = count_frames(out)
    if got != slot.frames:
        raise PipelineError(f"{slot.clip_id}: 세그먼트 프레임 수 {got} ≠ 계획 {slot.frames} "
                            f"(원본 {slot.source} {slot.src_start:.3f}초~). 장면 끝이 파일 끝을 넘었을 수 있습니다.")


def count_frames(path: Path) -> int:
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                           "-show_entries", "stream=nb_read_packets", "-of", "json", str(path)],
                          capture_output=True, text=True)
    try:
        return int(json.loads(proc.stdout)["streams"][0]["nb_read_packets"])
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        raise PipelineError(f"프레임 수를 읽을 수 없습니다: {path}") from e


def audio_filter(cfg: dict, total: float, n_in: int) -> tuple[list[str], list[str]]:
    """(추가 입력 인자, 필터 체인). 출력 라벨은 [aout], 길이는 정확히 total초."""
    au = cfg.get("audio", {})
    music = resolve(cfg, au["music_path"]) if au.get("music_path") else None
    fmt = f"aformat=sample_rates={AUDIO_RATE}:channel_layouts=stereo"
    if not music:
        return [], [f"anullsrc=r={AUDIO_RATE}:cl=stereo,atrim=0:{total:.6f}[aout]"]
    if not music.is_file():
        raise PipelineError(f"음원 파일이 없습니다: {music} (config audio.music_path)")
    fade_out = max(total - 0.6, 0)
    return (["-ss", f"{float(au.get('music_start', 0)):.3f}", "-i", str(music)],
            [f"[{n_in}:a]{fmt},atrim=0:{total:.6f},asetpts=N/SR/TB,volume={au.get('music_volume', 0.9)},"
             f"afade=t=in:d=0.15,afade=t=out:st={fade_out:.3f}:d=0.6,"
             f"apad=whole_dur={total:.6f},atrim=0:{total:.6f}[aout]"])   # 음악이 짧으면 무음으로 채움


def compose(plan: EditPlan, lay: Layout, cfg: dict, work: Path, out: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    fps = plan.fps
    tcfg = styled_cfg(cfg, plan.style)          # 스타일의 글자 모양 (오디오 정책은 스타일과 무관)
    total_frames = plan.total_frames
    total = total_frames / fps

    # 1) 컷별 세그먼트 → concat (이번 실행 전용 폴더라 다른 작업의 세그먼트가 섞이지 않음)
    seg_list = work / "segments.txt"
    lines = []
    for k, cut in enumerate(plan.cuts):
        seg = work / f"seg_{k:02d}.mp4"
        _render_segment(cut, seg, lay, fps)
        lines.append(f"file '{seg.resolve().as_posix()}'")
    seg_list.write_text("\n".join(lines), encoding="utf-8")
    body = work / "body.mp4"
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(seg_list), "-c", "copy", str(body)], "concat")

    # 2) 제목/자막 PNG (안 들어가는 문구는 TextFitError — 자르지 않음)
    title_png = render_title(plan.title[0], plan.title[1], tcfg, lay, work / "title.png")
    sub_pngs = [render_subtitle(c.text, tcfg, lay, work / f"sub_{k:02d}.png") for k, c in enumerate(plan.captions)]

    # 3) 필터 그래프: 배경 → 영상 영역 → 제목 → 자막(자막 타임라인의 프레임 구간마다)
    inputs = ["-i", str(body), "-i", str(title_png)]
    for p in sub_pngs:
        inputs += ["-i", str(p)]
    fc = [f"color=c=black:s={lay.width}x{lay.height}:r={fps}:d={total:.6f}[bg]",
          f"[bg][0:v]overlay=0:{lay.video_y}:eof_action=pass[v0]",
          "[v0][1:v]overlay=0:0[v1]"]
    last = "v1"
    for k, c in enumerate(plan.captions):
        f0, f1 = plan.caption_frames(c)
        fc.append(f"[{last}][{k + 2}:v]overlay=0:0:enable='between(n,{f0},{f1 - 1})'[s{k}]")
        last = f"s{k}"

    extra, afc = audio_filter(cfg, total, 2 + len(sub_pngs))
    inputs += extra
    fc += afc
    if not extra:
        log.info("기본 출력은 무음입니다 (원본 소리·효과음 없음). 업로드 시 음악을 추가할 수 있습니다.")

    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([*inputs, "-filter_complex", ";".join(fc), "-map", f"[{last}]", "-map", "[aout]",
                "-frames:v", str(total_frames), "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-r", str(fps),
                "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-movflags", "+faststart", str(out)],
               "최종 합성")
    log.info("완성: %s (%.2f초, %d프레임, 컷 %d개, 자막 %d개)", out, total, total_frames, len(plan.cuts),
             len(plan.captions))
    return out
