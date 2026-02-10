# UV 사용법

* 레포지토리를 클론한 후 루트 디렉토리에서 `uv sync` 명령어 실행하면 가상환경이 생성됩니다.
* `uv run python run_pipeline.py` 실행 후 `uv run streamlit run streamlit.py` 실행하면 1차 목표였던 출력을 볼 수 있습니다.
* 패키지가 추가/삭제는 `uv add {패키지명}`, `uv remove {패키지명}`을 사용하시고 `uv lock` 명령어를 통해 디펜던시를 락하면 됩니다.

# 작업 흐름
* 모델 코드 작업은 기본적으로 브랜치 파서 `./app/` 하위 디렉토리에서 진행하고 pr 요청보내시면 됩니다.