"""
마켓 레포트에서 '종합 의견' (SELL/HOLD/BUY)을 추출하는 파서 클래스
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MarketReportParser:
    """마켓 레포트 텍스트에서 종합 의견을 추출하는 파서.

    기존 get_agent_opinion 함수 로직을 기반으로,
    파일 경로 또는 문자열 입력을 모두 지원합니다.

    Attributes:
        VALID_OPINIONS: 유효한 의견 값 집합 (SELL, HOLD, BUY)
    """

    VALID_OPINIONS = {"SELL", "HOLD", "BUY"}

    # 패턴 1: 테이블 헤더 + 구분선 + 데이터 행에서 종합 의견 추출
    _TABLE_PATTERN = re.compile(
        r"\|\s*어제 종가\s*\|[^|]*\|[^|]*\|[^|]*\|\s*종합 의견\s*\|"
        r".*?\n[^\n]*\n\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*(SELL|HOLD|BUY)\s*\|",
        re.DOTALL | re.MULTILINE,
    )

    # 패턴 2: '종합 의견' 근처에서 직접 SELL/HOLD/BUY 매칭
    _SIMPLE_PATTERN = re.compile(
        r"종합\s*의견[^|]*\|.*?\|\s*(SELL|HOLD|BUY)\s*\|",
        re.DOTALL,
    )

    # 테이블 구분선 패턴
    _SEPARATOR_PATTERN = re.compile(r"^\s*\|[\s:|\-]+\|\s*$")

    def parse_from_file(self, file_path: str) -> str:
        """파일 경로에서 레포트를 읽어 종합 의견을 추출합니다.

        Args:
            file_path: 레포트 파일의 경로 (절대/상대 경로 모두 가능)

        Returns:
            추출된 의견 (SELL, HOLD, BUY 중 하나).
            파일을 읽을 수 없거나 의견을 찾지 못한 경우 빈 문자열 반환.

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            IsADirectoryError: 경로가 디렉토리인 경우
        """
        path = Path(file_path)
        text = path.read_text(encoding="utf-8")
        return self.parse_from_text(text)

    def parse_from_text(self, text: str) -> str:
        """문자열에서 종합 의견을 추출합니다.

        테이블 형식의 헤더를 먼저 찾고, 실패하면 라인별 탐색으로 폴백합니다.

        Args:
            text: 파싱할 레포트 텍스트

        Returns:
            추출된 의견 (SELL, HOLD, BUY 중 하나).
            찾지 못한 경우 빈 문자열 반환.
        """
        if not text:
            logger.warning("빈 텍스트가 입력되었습니다.")
            return ""

        # 전략 1: 정규식으로 테이블 전체 매칭
        opinion = self._extract_by_table_pattern(text)
        if opinion:
            return opinion

        # 전략 2: 라인별 탐색
        opinion = self._extract_by_line_scan(text)
        if opinion:
            return opinion

        logger.warning("텍스트에서 '종합 의견' 값을 찾을 수 없습니다.")
        return ""

    def _extract_by_table_pattern(self, text: str) -> str:
        """정규식 패턴으로 테이블에서 종합 의견을 추출합니다."""
        matches = self._TABLE_PATTERN.findall(text)
        if matches:
            opinion = matches[0].strip()
            if opinion in self.VALID_OPINIONS:
                logger.info(f"종합 의견 추출 성공 (테이블 패턴): {opinion}")
                return opinion
            logger.warning(f"추출된 값이 유효하지 않음: {opinion}")
        return ""

    def _extract_by_line_scan(self, text: str) -> str:
        """라인별 탐색으로 종합 의견을 추출합니다."""
        lines = text.split("\n")

        for i, line in enumerate(lines):
            if "종합 의견" not in line or "|" not in line:
                continue

            # 헤더 아래 최대 5줄에서 데이터 행 탐색
            for j in range(i + 1, min(i + 6, len(lines))):
                data_line = lines[j]

                # 구분선 건너뛰기
                if self._SEPARATOR_PATTERN.match(data_line):
                    continue

                if "|" not in data_line:
                    continue

                cells = [cell.strip() for cell in data_line.split("|") if cell.strip()]
                if not cells:
                    continue

                last_cell = cells[-1]
                for valid_opinion in self.VALID_OPINIONS:
                    if valid_opinion in last_cell:
                        logger.info(f"종합 의견 추출 성공 (라인 탐색): {valid_opinion}")
                        return valid_opinion

            # 헤더를 찾았지만 데이터를 못 찾으면 종료
            break

        return ""
