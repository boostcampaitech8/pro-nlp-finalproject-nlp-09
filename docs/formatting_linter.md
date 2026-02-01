# 포맷팅 팁, 메모 (에디터 외 로컬 포맷팅, 린팅)

# 질문 2.

User: ruff 포매터를 app 디렉토리의 모든 코드에 한번에 적용할 수 있는 방법 있어?

# 답변 2.

이제 루트에서 다음 명령어로 사용할 수 있습니다:

```bash
# 포매팅 (자동 수정)
ruff format app/

# 린트 체크
ruff check app/

# 린트 + 자동 수정
ruff check app/ --fix

# 전체 (format + lint fix)
ruff format app/ && ruff check app/ --fix
```

**설정 요약:**

| 설정 | 값 |
|------|-----|
| 대상 | app, run_pipeline.py, streamlit.py |
| Python 버전 | 3.11 |
| 라인 길이 | 120자 |
| 따옴표 | 쌍따옴표 (`"`) |
| 린트 규칙 | E, W, F, I (isort), B, UP |