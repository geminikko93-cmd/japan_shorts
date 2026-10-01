"""완성 영상 검증: python tests/verify_output.py <영상.mp4> <edit_plan.json> [프레임 저장 폴더]

- 1080x1920 / 30fps / 영상·오디오 길이 일치 (AAC 1프레임 = 23ms 허용)
- 오디오를 디코딩해 실제 샘플 값 확인 (BGM 미설정이면 전부 0이어야 함)
- 컷마다 출력 첫·끝 프레임이 계획한 원본 구간(크롭·스케일 적용)의 프레임과 일치하는지 비교
- 시작·중간·끝 프레임 PNG 저장 (사람이 제목/자막 위치·얼굴 잘림을 눈으로 확인)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shorts.plan import load_plan  # noqa: E402

SMALL = (108, 135)   # 비교용 축소 크기 (영상 영역 4:5 기준)


def run(args: list[str]) -> bytes:
    return subprocess.run(args, capture_output=True, check=True).stdout


def probe(path: Path) -> dict:
    out = json.loads(run(["ffprobe", "-v", "error", "-count_packets", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,duration,nb_read_packets",
                          "-of", "json", str(path)]))
    return {s["codec_type"]: s for s in out["streams"]}


def gray(frames: bytes, w: int, h: int) -> np.ndarray:
    return np.frombuffer(frames, np.uint8).reshape(-1, h, w).astype(np.float32)


def out_frames(video: Path, y: int, vh: int, w: int, idx: list[int]) -> np.ndarray:
    """출력 프레임 idx(순서 그대로)의 영상 영역. select는 시간 순으로 내보내므로 정렬 후 다시 매핑."""
    uniq = sorted(set(idx))
    sel = "+".join(f"eq(n\\,{i})" for i in uniq)
    raw = run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
               f"select='{sel}',crop={w}:{vh}:0:{y},scale={SMALL[0]}:{SMALL[1]}:flags=area,format=gray",
               "-fps_mode", "passthrough", "-f", "rawvideo", "-"])
    frames = gray(raw, *SMALL)
    assert len(frames) == len(uniq), (len(frames), len(uniq))
    pos = {n: k for k, n in enumerate(uniq)}
    return np.stack([frames[pos[i]] for i in idx])


def src_frame(src: str, t: float, crop, w: int, vh: int) -> np.ndarray:
    return src_window(src, t, t + 0.001, crop, w, vh)[1][0]


def src_window(src: str, t0: float, t1: float, crop, w: int, vh: int) -> tuple[list[float], np.ndarray]:
    """원본 [t0, t1] 구간의 모든 프레임 (원래 타임스탬프, 크롭·축소 이미지)."""
    import re
    x, y, cw, ch = crop
    t0 = max(t0, 0.0)
    proc = subprocess.run(["ffmpeg", "-v", "info", "-hide_banner", "-copyts", "-ss", f"{t0:.4f}",
                           "-t", f"{max(t1 - t0, 0.001):.4f}", "-i", src, "-vf",
                           f"crop={cw}:{ch}:{x}:{y},scale={w}:{vh}:flags=lanczos,"
                           f"scale={SMALL[0]}:{SMALL[1]}:flags=area,format=gray,showinfo",
                           "-fps_mode", "passthrough", "-f", "rawvideo", "-"], capture_output=True, check=True)
    ts = [float(m) for m in re.findall(rb"pts_time:([0-9.]+)", proc.stderr)]
    frames = gray(proc.stdout, *SMALL)
    return ts[:len(frames)], frames


def match(frame: np.ndarray, ts: list[float], cands: np.ndarray, lo: float, hi: float) -> tuple[float, float, int]:
    """가장 비슷한 원본 프레임 (타임스탬프, 밝기 차, 동률 개수).

    정지 화면처럼 똑같은 프레임이 여러 개면(차이 0.3 이내) 픽셀로는 구분할 수 없으므로,
    그중 계획 구간 [lo, hi) 안에 있는 것을 고르고 동률 개수를 함께 보고한다.
    """
    d = np.abs(cands - frame).mean(axis=(1, 2))
    best = float(d.min())
    ties = [k for k in range(len(d)) if d[k] <= best + 0.3]
    inside = [k for k in ties if lo - 1e-3 <= ts[k] < hi + 1e-3]
    k = inside[0] if inside else ties[0]
    return ts[k], float(d[k]), len(ties)


def main() -> int:
    video, plan_path = Path(sys.argv[1]), Path(sys.argv[2])
    shots = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    plan = load_plan(plan_path)
    cfg_w, fps = 1080, plan.fps
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pre = cfg["layout"]["presets"][plan.layout]
    vy, vh = pre["title_h"], pre["video_h"]
    ok = True

    p = probe(video)
    v, a = p["video"], p.get("audio")
    n = int(v["nb_read_packets"])
    want = plan.total_frames
    vd, ad = float(v["duration"]), float(a["duration"]) if a else -1
    print(f"[형식] {v['codec_name']} {v['width']}x{v['height']} {v['r_frame_rate']} 프레임 {n} (계획 {want}) / "
          f"오디오 {a['codec_name'] if a else '없음'}")
    ok &= (v["width"], v["height"], v["r_frame_rate"], n) == (1080, 1920, "30/1", want)
    print(f"[길이] 영상 {vd:.4f}s / 오디오 {ad:.4f}s / 계획 {plan.total:.4f}s / 차이 {abs(vd - ad) * 1000:.1f}ms "
          f"(허용 {1024 / 44100 * 1000:.1f}ms)")
    ok &= abs(vd - ad) <= 1024 / 44100 + 1e-3 and abs(vd - plan.total) <= 1 / fps

    pcm = np.frombuffer(run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a:0", "-f", "s16le",
                             "-ac", "2", "-ar", "44100", "-"]), np.int16)
    peak = int(np.abs(pcm).max()) if pcm.size else -1
    print(f"[오디오] 디코딩 샘플 {pcm.size // 2}개 ({pcm.size / 2 / 44100:.3f}s), 최대 절댓값 {peak}"
          + (" → 완전 무음" if peak == 0 else ""))
    if not cfg["audio"].get("music_path"):
        ok &= peak == 0

    # 컷별 프레임 일치
    f0, worst = 0, 0.0
    firsts, lasts = [], []
    for s in plan.cuts:
        firsts.append(f0)
        lasts.append(f0 + s.frames - 1)
        f0 += s.frames
    got = out_frames(video, vy, vh, cfg_w, firsts + lasts)
    gf, gl = got[:len(firsts)], got[len(firsts):]
    # 구간 경계 바깥 0.2초까지 포함한 원본 프레임 중 가장 비슷한 것을 찾아, 그 시각이 계획 구간 안인지 확인
    print("[컷] 번호 장면 | 계획 구간(초) | 출력 첫/끝 프레임과 가장 비슷한 원본 프레임 시각 (밝기 차) | 판정")
    margin = 0.2
    for k, s in enumerate(plan.cuts):
        end = s.src_start + s.frames / fps
        ts0, win0 = src_window(s.source, s.src_start - margin, s.src_start + margin, s.crop, cfg_w, vh)
        ts1, win1 = src_window(s.source, end - margin, end + margin, s.crop, cfg_w, vh)
        t_first, d0, n0 = match(gf[k], ts0, win0, s.src_start, end)
        t_last, d1, n1 = match(gl[k], ts1, win1, s.src_start, end)
        inside = s.src_start - 1e-3 <= t_first and t_last < end + 1e-3
        good = inside and max(d0, d1) < 6
        worst = max(worst, d0, d1)
        ok &= good
        print(f"  #{k + 1:<2} {s.clip_id:<6}| {s.src_start:7.3f}~{end:7.3f} | 첫 {t_first:7.3f} ({d0:4.1f}) "
              f"끝 {t_last:7.3f} ({d1:4.1f}) | {'구간 안' if inside else '구간 밖!'}{'' if good else ' ✗'}"
              f"{' (정지 화면: 동일 프레임 %d개)' % max(n0, n1) if max(n0, n1) > 1 else ''}  {s.role}")
    print(f"[컷] 최대 밝기 차 {worst:.2f} (같은 프레임 기준 < 6)")

    # 자막 타임라인: 계획상 자막이 있는 프레임에만 자막 영역에 글자가 있어야 함
    ch = 1920 - vy - vh
    raw = run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
               f"crop={cfg_w}:{ch}:0:{vy + vh},scale=108:{ch // 10},format=gray", "-f", "rawvideo", "-"])
    px = [int((a > 128).sum()) for a in np.frombuffer(raw, np.uint8).reshape(-1, ch // 10, 108)]
    has = [False] * want
    for c in plan.captions:
        f0_, f1_ = plan.caption_frames(c)
        for i in range(f0_, f1_):
            has[i] = True
    wrong_on = [i for i in range(want) if not has[i] and px[i] > 0]
    wrong_off = [i for i in range(want) if has[i] and px[i] < 30]
    print(f"[자막] 자막 있는 프레임 {sum(has)} / 없는 프레임 {want - sum(has)} — 계획과 다른 프레임: "
          f"자막이 생긴 곳 {len(wrong_on)}, 사라진 곳 {len(wrong_off)}")
    for c in plan.captions:
        f0_, f1_ = plan.caption_frames(c)
        print(f"  {c.id} {f0_ / fps:6.2f}~{f1_ / fps:6.2f}s (컷 {c.first_cut + 1}-{c.last_cut + 1}) {c.text}")
    ok &= not wrong_on and not wrong_off

    if shots:
        shots.mkdir(parents=True, exist_ok=True)
        for name, i in (("start", 0), ("mid", want // 2), ("end", want - 1)):
            run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select='eq(n\\,{i})'", "-fps_mode", "passthrough",
                 "-frames:v", "1", str(shots / f"{video.stem}_{name}.png")])
        print(f"[프레임] {shots} 에 start/mid/end 저장")
    print("결과:", "통과" if ok else "실패")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
