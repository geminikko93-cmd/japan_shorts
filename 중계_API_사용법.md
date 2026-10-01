# 중계 서버 연결 방법

1. 압축을 새 폴더에 풀고 `japan_shorts` 폴더에서 PowerShell을 엽니다.
2. `python -m pip install -r requirements.txt`를 실행합니다.
3. `.env.example`을 `.env`로 복사합니다. 기존 프로그램 폴더에 적용할 때는 기존 `.env`를 덮지 말고 아래 Claude 설정을 추가/수정하세요.

```ini
ANTHROPIC_AUTH_TOKEN=발급받은_중계_API_키
ANTHROPIC_BASE_URL=https://codex.hungnguyen.codes
ANTHROPIC_MODEL=claude-opus-5-5
YOUTUBE_API_KEY=기존_YouTube_API_키
```

4. `run_ui.bat`을 더블클릭하거나 `python -m streamlit run app.py`를 실행합니다.
5. `② 칭찬글 · BGM 추천`에서 인물을 입력하고 기획안을 생성하면 실제 API 연결을 확인할 수 있습니다. API 사용량이 발생합니다. 기존에 저장된 기획안을 불러오는 것만으로는 연결을 확인할 수 없습니다.

추천, 기획안 생성, 한국어 번역에 동일한 중계 설정을 사용합니다.
Claude Code 설치와 `.claude/settings.json`은 이 프로그램 실행에 필요하지 않습니다.
키나 모델을 변경하면 Streamlit 서버를 종료한 후 다시 실행하세요.
사이드바의 키 표시는 설정 여부이며 연결 성공 표시가 아닙니다.

## 호환성

표준 Anthropic Messages API의 Bearer 토큰 인증을 사용합니다.
중계 서버에서는 ANTHROPIC_AUTH_TOKEN이 필수이며 기존 ANTHROPIC_API_KEY를 대신 전송하지 않습니다.
모델 ID는 서버에서 실제 지원해야 합니다. 운영자가 다른 ID를 안내하면 ANTHROPIC_MODEL을 변경하세요.
기본 json_mode: prompt는 JSON 스키마를 프롬프트에 넣고 응답을 로컬에서 검증합니다.
서버가 output_config를 지원할 때 config.yaml의 llm.json_mode를 schema로 변경할 수 있습니다.
effort 설정은 schema 모드에서만 전송됩니다. 베타 헤더와 서버 측 fallback 옵션은 전송하지 않습니다.
자동 재요청은 하지 않습니다. 잘못된 JSON, 응답 잘림, 인증/연결 오류는 화면에 표시됩니다.

배포 ZIP에는 원본 .env, 업로드 인증 파일, 임시 작업물과 생성 영상이 포함되지 않습니다.
기존 API 키는 원본 .env에서 직접 옮겨 넣으세요. 실제 서버 호출은 발급받은 키로 실행해야 검증됩니다.
