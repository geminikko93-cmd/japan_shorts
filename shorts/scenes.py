"""③ 장면 분석: 장면 분할 + 얼굴 검출 + 대표 프레임.

캡컷 수동 작업 대체:
- '장면 분할'           → PySceneDetect ContentDetector. 원래 장면 경계를 그대로 보존한다
                           (웃기 직전과 웃는 순간을 억지로 나누지 않음. 편집 구간은 계획 단계에서 장면 안에서 고름)
- 'S로 미리보고 D로 삭제' → 얼굴이 충분히 크게 나온 장면만 남김 (YuNet 얼굴 검출)
- 장면 점수              → 얼굴 높이 비율 × 3개 샘플 중 검출 비율. 즉 '얼굴이 크고 안정적으로 보이는가'일 뿐
                           예쁜 장면·웃는 장면을 판별하지 않는다 → 첫 컷은 편집 화면에서 사람이 고른다
- 다른 사람 장면 제거     → SFace 얼굴 임베딩으로 '가장 많이 나온 얼굴'만 남김. 휴리스틱이라
                           검색한 인물이 맞는지 보장하지 않는다 (소스에 다른 인물이 더 많이 나오면 틀림)
- 반복 장면 판별용        → 대표 프레임의 지각 해시(dHash). 인물 유사도(SFace)와는 별개
크롭 위치는 레이아웃마다 다르므로 여기서 정하지 않고 얼굴 박스만 저장한다 (layout.crop_for).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import PipelineError, log
from .sources import Source

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")
SAME_PERSON_COS = 0.363  # SFace 공식 cosine 임계값 (이상이면 동일 인물)
MIN_SCENE_SEC = 1.0      # 이보다 짧은 장면은 버림 (컷 최소 길이보다 짧아 쓸 수 없음)
WINDOW_SEC = 6.0         # 긴 연속 장면 안의 후보 구간 길이 (장면 경계는 그대로, 고를 수 있는 순간만 여러 개)
THUMB_SIDE = 480         # 대표 프레임 저장 크기(긴 변)
DETECT_MAX_SIDE = 640    # 얼굴 검출용 축소 크기(긴 변). 얼빡 컷 대상 얼굴은 이 정도면 충분히 잡힘


@dataclass
class Clip:
    """장면 안의 후보 구간. 짧은 장면은 장면 전체, 긴 연속 장면은 WINDOW_SEC 정도의 후보 여러 개.

    start/end = 후보 구간(대표 프레임·얼굴·해시의 기준), scene_start/scene_end = 원래 장면 경계.
    컷은 장면 안이라면 후보 구간 밖으로도 늘릴 수 있지만 장면 경계는 넘지 않는다.
    """
    id: str                   # 'S1-03' = 1번째 소스의 3번째 장면, 'S1-03b' = 그 장면의 2번째 후보 구간
    source: Source
    start: float
    end: float                # 파일 끝을 넘지 않도록 잘린 값
    score: float              # 얼굴 높이 비율 × 검출 안정성 (0~1). 미모/표정 점수가 아님
    face: tuple[int, int, int, int] | None   # 원본 좌표 얼굴 박스 (x, y, w, h)
    frame_size: tuple[int, int]              # 원본 (폭, 높이)
    fps: float = 30.0         # 원본 프레임레이트 (시작 시각을 원본 프레임 격자에 맞추는 데 사용)
    dhash: int = 0            # 대표 프레임 64bit 지각 해시 (시각적 중복 판별)
    thumb: str = ""           # 대표 프레임 JPG 경로 (THUMB_SIDE로 축소)
    feat: np.ndarray | None = None  # 얼굴 임베딩 (동일 인물 필터 전용)
    scene_start: float | None = None   # 원래 장면 경계 (없으면 start/end와 같음)
    scene_end: float | None = None

    def __post_init__(self):
        if self.scene_start is None:
            self.scene_start = self.start
        if self.scene_end is None:
            self.scene_end = self.end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def scene_duration(self) -> float:
        return self.scene_end - self.scene_start

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source.path, "start": self.start, "end": self.end,
                "score": round(self.score, 4), "face": list(self.face) if self.face else None,
                "frame_size": list(self.frame_size), "fps": self.fps, "dhash": f"{self.dhash:016x}", "thumb": self.thumb,
                "scene_start": self.scene_start, "scene_end": self.scene_end}


def clip_from_dict(d: dict, sources_by_path: dict[str, Source]) -> Clip:
    """analysis.json 항목 → Clip (얼굴 임베딩은 저장하지 않으므로 None)."""
    src = sources_by_path.get(d["source"]) or Source(path=d["source"])
    return Clip(d["id"], src, float(d["start"]), float(d["end"]), float(d["score"]),
                tuple(d["face"]) if d.get("face") else None, tuple(d["frame_size"]), float(d.get("fps", 30.0)),
                int(d.get("dhash", "0"), 16), d.get("thumb", ""), None,
                d.get("scene_start"), d.get("scene_end"))


def dhash(frame: np.ndarray) -> int:
    g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class FaceDetector:
    """YuNet(정확) → Haar(대체) → 없음(중앙 크롭) 순으로 사용."""

    def __init__(self, model_dir: Path):
        self.kind = "none"
        self.yunet = None
        self.haar = None
        model = model_dir / "face_detection_yunet_2023mar.onnx"
        try:
            if not model.exists():
                _download(YUNET_URL, model)
            self.yunet = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.75, 0.3, 50)
            self.kind = "yunet"
        except Exception as e:
            log.warning("YuNet 사용 불가 (%s) → Haar 시도", e)
            data = getattr(cv2, "data", None)
            casc = Path(data.haarcascades) / "haarcascade_frontalface_default.xml" if data else None
            if casc and casc.exists():
                self.haar = cv2.CascadeClassifier(str(casc))
                self.kind = "haar"
        if self.kind == "none":
            log.warning("얼굴 검출기를 쓸 수 없어 중앙 크롭으로 진행합니다.")

    def largest_face(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray | None] | None:
        """(x, y, w, h), YuNet 원본 행(랜드마크 포함, 얼굴 인식용 / Haar면 None)."""
        h, w = frame.shape[:2]
        if self.yunet is not None:
            k = min(1.0, DETECT_MAX_SIDE / max(w, h))   # 1080p 원본 그대로 검출하면 수 배 느림
            small = cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame
            self.yunet.setInputSize((small.shape[1], small.shape[0]))
            _, faces = self.yunet.detect(small)
            if faces is None or len(faces) == 0:
                return None
            row = max(faces, key=lambda f: f[2] * f[3]).copy()
            row[:14] /= k                                 # 박스+랜드마크를 원본 좌표로
            x, y, fw, fh = row[:4]
            return (int(x), int(y), int(fw), int(fh)), row
        if self.haar is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces) == 0:
                return None
            return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3])), None
        return None


def detect_scenes(path: str, threshold: float) -> list[tuple[float, float]]:
    from scenedetect import ContentDetector, detect

    def secs(tc) -> float:
        return float(tc.get_seconds() if hasattr(tc, "get_seconds") else tc.seconds)

    try:
        raw = detect(path, ContentDetector(threshold=threshold))
    except Exception as e:
        raise PipelineError(f"장면 분할 실패: {path} ({e})") from e
    scenes = [(secs(a), secs(b)) for a, b in raw]
    if not scenes:  # 컷 전환이 없는 영상 → 전체를 하나의 장면으로
        cap = cv2.VideoCapture(path)
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()
        scenes = [(0.0, n / fps)]
    return [(a, b) for a, b in scenes if b - a >= MIN_SCENE_SEC]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("모델 다운로드: %s", url)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


class FaceRecognizer:
    """SFace 임베딩. 모델을 못 쓰면 None을 돌려주고 필터를 건너뜀."""

    def __init__(self, model_dir: Path):
        self.sf = None
        model = model_dir / "face_recognition_sface_2021dec.onnx"
        try:
            if not model.exists():
                _download(SFACE_URL, model)
            self.sf = cv2.FaceRecognizerSF.create(str(model), "")
        except Exception as e:
            log.warning("얼굴 인식 모델 사용 불가 (%s) → 동일 인물 필터 생략", e)

    def embed(self, frame: np.ndarray, row: np.ndarray) -> np.ndarray | None:
        if self.sf is None:
            return None
        try:
            f = self.sf.feature(self.sf.alignCrop(frame, row)).flatten()
        except cv2.error:
            return None
        return f / (np.linalg.norm(f) or 1.0)


def keep_main_person(clips: list[Clip]) -> list[Clip]:
    """가장 많이 등장하는 얼굴과 다른 사람이 나오는 장면을 제거.

    '가장 많이 나온 얼굴 = 검색한 주인공'이라는 가정에 기댄 휴리스틱이다. 대상 인물 확인을 보장하지 않는다.
    """
    have = [c for c in clips if c.feat is not None]
    if len(have) < 3:
        return clips
    F = np.stack([c.feat for c in have])
    sim = F @ F.T
    anchor = int(np.argmax((sim >= SAME_PERSON_COS).sum(1)))
    group = F[sim[anchor] >= SAME_PERSON_COS]
    center = group.mean(0)
    center /= np.linalg.norm(center) or 1.0
    kept = [c for c in clips if c.feat is None or float(c.feat @ center) >= SAME_PERSON_COS]
    log.info("동일 인물 필터: 장면 %d개 → %d개 (다른 인물 %d개 제거)", len(clips), len(kept), len(clips) - len(kept))
    return kept or clips


def analyze(sources: list[Source], cfg: dict, model_dir: Path, thumb_dir: Path | None = None,
            info: dict | None = None) -> list[Clip]:
    """모든 소스에서 쓸 수 있는 장면을 고르고 점수순(+소스 교차)으로 반환.

    info에 실제 사용한 검출 모드와 생략된 기능을 기록한다 (UI에 그대로 표시).
    """
    from .common import probe_duration

    lay = cfg["layout"]
    fps = float(cfg["video"]["fps"])
    det = FaceDetector(model_dir)
    rec = FaceRecognizer(model_dir) if det.kind == "yunet" else None
    if thumb_dir:
        thumb_dir.mkdir(parents=True, exist_ok=True)
    info = info if info is not None else {}
    info.update({"detector": det.kind, "person_filter": bool(rec and rec.sf is not None), "skipped": []})
    if det.kind == "haar":
        info["skipped"].append("YuNet 대신 Haar 검출기 사용 (정확도 낮음)")
    if det.kind == "none":
        info["skipped"].append("얼굴 검출 없음: 모든 장면을 중앙 크롭, 얼굴 장면 선별 안 함")
    if not info["person_filter"]:
        info["skipped"].append("얼굴 인식 모델 없음: 동일 인물 필터 생략")
    info["per_source"] = {}
    clips: list[Clip] = []
    for si, src in enumerate(sources, 1):
        cap = cv2.VideoCapture(src.path)
        if not cap.isOpened():
            log.warning("영상을 열 수 없어 건너뜀: %s", src.path)
            continue
        fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        # 장면 끝을 실제 파일 길이 안쪽으로 (마지막 프레임 2개 여유) → 파일 끝 패딩에 의존하지 않음
        file_end = probe_duration(Path(src.path)) - 2.0 / min(src_fps, fps)
        scenes = [(a, min(b, file_end)) for a, b in detect_scenes(src.path, cfg["sources"]["scene_threshold"])
                  if min(b, file_end) - a >= MIN_SCENE_SEC]
        kept = 0
        windows = []
        for k, (sa, sb) in enumerate(scenes, 1):
            n = int(np.floor((sb - sa) / WINDOW_SEC)) if sb - sa >= 1.5 * WINDOW_SEC else 1
            step = (sb - sa) / n
            for j in range(n):
                wid = f"S{si}-{k:02d}" + (chr(ord("a") + j) if n > 1 and j < 26 else (str(j) if n > 1 else ""))
                windows.append((wid, sa + j * step, sa + (j + 1) * step if j < n - 1 else sb, sa, sb))
        for cid, a, b, sa, sb in windows:
            faces, best, mid = [], None, None   # best = (얼굴 높이, 프레임, YuNet 행) → 임베딩용
            for r in (0.25, 0.5, 0.75):
                cap.set(cv2.CAP_PROP_POS_MSEC, (a + (b - a) * r) * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                if r == 0.5 or mid is None:
                    mid = frame
                found = det.largest_face(frame)
                if found is not None:
                    f, row = found
                    faces.append(f)
                    if row is not None and (best is None or f[3] > best[0]):
                        best = (f[3], frame, row)
            if mid is None:
                continue
            if faces:
                face = tuple(int(np.median([f[i] for f in faces])) for i in range(4))
                score = face[3] / fh * (len(faces) / 3)   # 얼굴 크기 × 등장 안정성
            else:
                face, score = None, 0.0
            if det.kind != "none" and score < lay["min_face_ratio"]:
                continue   # 얼굴이 안 나오거나 너무 작은 장면은 버림
            thumb = ""
            if thumb_dir:
                kf = THUMB_SIDE / max(fw, fh)
                small = cv2.resize(mid, (round(fw * kf), round(fh * kf)), interpolation=cv2.INTER_AREA)
                thumb = str(thumb_dir / f"{cid}.jpg")
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise PipelineError(f"썸네일 저장 실패: {thumb}")
                Path(thumb).write_bytes(buf.tobytes())   # cv2.imwrite는 Windows 한글/일본어 경로에서 조용히 실패
            feat = rec.embed(best[1], best[2]) if rec and best else None
            clips.append(Clip(cid, src, float(np.ceil(a * 1e4) / 1e4), float(np.floor(b * 1e4) / 1e4), score, face,
                              (fw, fh), float(src_fps), dhash(mid), thumb, feat,
                              float(np.ceil(sa * 1e4) / 1e4), float(np.floor(sb * 1e4) / 1e4)))
            kept += 1
        cap.release()
        log.info("%s: 장면 %d개 (후보 구간 %d개) 중 %d개 사용 (검출기=%s)", Path(src.path).name, len(scenes),
                 len(windows), kept, det.kind)
        info["per_source"][src.path] = {"scenes": len(scenes), "windows": len(windows), "kept": kept}

    if not clips:
        raise PipelineError("얼굴이 충분히 나온 장면이 없습니다. 다른 소스 영상을 사용하세요.")
    kept_main = keep_main_person(clips)
    info["person_filter_removed"] = len(clips) - len(kept_main)
    return order_clips(kept_main)


def order_clips(clips: list[Clip]) -> list[Clip]:
    """얼굴 점수 최고 장면을 맨 앞에(사람이 첫 컷을 바꿀 수 있음), 나머지는 소스를 번갈아가며 점수순."""
    best = max(clips, key=lambda c: c.score)
    rest = [c for c in clips if c is not best]
    by_src: dict[str, list[Clip]] = {}
    for c in sorted(rest, key=lambda c: -c.score):
        by_src.setdefault(c.source.path, []).append(c)
    ordered = [best]
    queues = list(by_src.values())
    while any(queues):
        for q in queues:
            if q:
                ordered.append(q.pop(0))
    return ordered
