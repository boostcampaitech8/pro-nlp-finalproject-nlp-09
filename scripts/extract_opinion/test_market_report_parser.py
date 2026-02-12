"""
MarketReportParser 클래스 테스트
"""

import os
import unittest
import tempfile
import logging

from market_report_parser import MarketReportParser

logging.getLogger("market_report_parser").setLevel(logging.WARNING)

# data/reports 디렉토리 경로
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reports")


class TestParseFromText(unittest.TestCase):
    """parse_from_text 메서드 테스트"""

    def setUp(self):
        self.parser = MarketReportParser()

    # --- 기본 SELL / HOLD / BUY ---

    def test_basic_sell(self):
        text = (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 441.5 | 435.06 | Down | 부정적 | SELL |\n"
        )
        self.assertEqual(self.parser.parse_from_text(text), "SELL")

    def test_basic_hold(self):
        text = (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 427.25 | 435.97 | Down | 중립적 | HOLD |\n"
        )
        self.assertEqual(self.parser.parse_from_text(text), "HOLD")

    def test_basic_buy(self):
        text = (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 535.75 | 541.09 | Up | 긍정적 | BUY |\n"
        )
        self.assertEqual(self.parser.parse_from_text(text), "BUY")

    # --- 빈 입력 / 의견 없음 ---

    def test_empty_string(self):
        self.assertEqual(self.parser.parse_from_text(""), "")

    def test_no_opinion_column(self):
        text = (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 |\n"
            "|:---:|:---:|:---:|:---:|\n"
            "| 535.75 | 541.09 | Up | 부정적 |\n"
        )
        self.assertEqual(self.parser.parse_from_text(text), "")

    def test_invalid_opinion_value(self):
        text = (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 535.75 | 541.09 | Up | 부정적 | WAIT |\n"
        )
        self.assertEqual(self.parser.parse_from_text(text), "")

    def test_no_table_at_all(self):
        text = "이것은 테이블이 없는 일반 텍스트입니다.\n아무 데이터도 없습니다."
        self.assertEqual(self.parser.parse_from_text(text), "")

    # --- 공백 / 서식 변형 ---

    def test_extra_whitespace(self):
        text = (
            "|  어제 종가  |  Prophet 예측  |  XGBoost 방향  |  뉴스 심리  |  종합 의견  |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "|  535.75  |  541.09  |  Up  |  부정적  |  HOLD  |\n"
        )
        self.assertEqual(self.parser.parse_from_text(text), "HOLD")

    def test_different_separator_style(self):
        text = (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|---------|--------------|--------------|-----------|----------|\n"
            "| 535.75 | 541.09 | Up | 부정적 | SELL |\n"
        )
        self.assertEqual(self.parser.parse_from_text(text), "SELL")

    # --- 여러 테이블 / 긴 컨텍스트 ---

    def test_multiple_tables_returns_first(self):
        text = (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 535.75 | 541.09 | Up | 긍정적 | BUY |\n"
            "\nSome other content...\n\n"
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 535.75 | 541.09 | Down | 부정적 | SELL |\n"
        )
        self.assertEqual(self.parser.parse_from_text(text), "BUY")

    def test_table_embedded_in_long_context(self):
        prefix = "무의미한 텍스트 라인\n" * 100
        suffix = "추가 내용\n" * 100
        table = (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 441.5 | 435.06 | Down | 부정적 | SELL |\n"
        )
        text = prefix + table + suffix
        self.assertEqual(self.parser.parse_from_text(text), "SELL")

    def test_table_with_markdown_header(self):
        """실제 레포트처럼 마크다운 헤더와 함께 있는 경우"""
        text = (
            "# [Daily Market Report] Corn\n"
            "**날짜**: 2025-11-14 | **종목**: 옥수수\n\n"
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 441.5 | 435.06 | Down | 부정적 | SELL |\n\n"
            "---\n\n### 1. 퀀트 기반 기술적 분석\n"
        )
        self.assertEqual(self.parser.parse_from_text(text), "SELL")

    def test_escaped_newlines_in_json_like_text(self):
        """JSON-like 포맷에서 이스케이프된 줄바꿈이 있는 경우"""
        text = (
            '[{\'type\': \'text\', \'text\': "**날짜**: 2025-11-14 | **종목**: wheat\\n\\n'
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\\n"
            "|:---:|:---:|:---:|:---:|:---:|\\n"
            '| 535.75 | 541.09 | Up | 부정적 | HOLD |\\n"}]'
        )
        self.assertEqual(self.parser.parse_from_text(text), "HOLD")

    # --- 성능 ---

    def test_large_text_performance(self):
        import time

        large_text = "Some random content\n" * 10000
        large_text += (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 535.75 | 541.09 | Up | 부정적 | HOLD |\n"
        )
        large_text += "More random content\n" * 10000

        start = time.time()
        result = self.parser.parse_from_text(large_text)
        elapsed = time.time() - start

        self.assertEqual(result, "HOLD")
        self.assertLess(elapsed, 1.0, f"처리 시간 초과: {elapsed:.2f}초")


class TestParseFromFile(unittest.TestCase):
    """parse_from_file 메서드 테스트"""

    def setUp(self):
        self.parser = MarketReportParser()

    # --- 실제 data/reports 파일 테스트 ---

    def _report_path(self, filename: str) -> str:
        return os.path.join(REPORTS_DIR, filename)

    def test_report_2025_11_14(self):
        path = self._report_path("market_report_2025-11-14.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "SELL")

    def test_report_2025_11_13(self):
        path = self._report_path("market_report_2025-11-13.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "HOLD")

    def test_report_2025_11_12(self):
        path = self._report_path("market_report_2025-11-12.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "SELL")

    def test_report_2025_11_11(self):
        path = self._report_path("market_report_2025-11-11.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "SELL")

    def test_report_2025_11_10(self):
        path = self._report_path("market_report_2025-11-10.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "HOLD")

    def test_report_2025_11_7(self):
        path = self._report_path("market_report_2025-11-7.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "SELL")

    def test_report_2025_11_6(self):
        path = self._report_path("market_report_2025-11-6.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "SELL")

    def test_report_2025_11_5(self):
        path = self._report_path("market_report_2025-11-5.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "SELL")

    def test_report_2025_11_4(self):
        path = self._report_path("market_report_2025-11-4.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "SELL")

    def test_report_2025_11_3(self):
        path = self._report_path("market_report_2025-11-3.txt")
        if not os.path.exists(path):
            self.skipTest(f"파일 없음: {path}")
        self.assertEqual(self.parser.parse_from_file(path), "HOLD")

    # --- 임시 파일 / 에러 케이스 ---

    def test_temp_file_with_buy(self):
        content = (
            "| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |\n"
            "|:---:|:---:|:---:|:---:|:---:|\n"
            "| 500.0 | 510.0 | Up | 긍정적 | BUY |\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            result = self.parser.parse_from_file(f.name)
        os.unlink(f.name)
        self.assertEqual(result, "BUY")

    def test_temp_file_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("")
            f.flush()
            result = self.parser.parse_from_file(f.name)
        os.unlink(f.name)
        self.assertEqual(result, "")

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.parser.parse_from_file("/nonexistent/path/report.txt")

    def test_directory_path_raises(self):
        with self.assertRaises(IsADirectoryError):
            self.parser.parse_from_file(REPORTS_DIR)


class TestAllReportsNonEmpty(unittest.TestCase):
    """data/reports 디렉토리의 모든 레포트에서 의견 추출이 되는지 일괄 검증"""

    def setUp(self):
        self.parser = MarketReportParser()

    def test_all_reports_return_valid_opinion(self):
        if not os.path.isdir(REPORTS_DIR):
            self.skipTest(f"디렉토리 없음: {REPORTS_DIR}")

        report_files = [
            f for f in os.listdir(REPORTS_DIR) if f.endswith(".txt")
        ]
        self.assertGreater(len(report_files), 0, "레포트 파일이 없습니다")

        for filename in report_files:
            path = os.path.join(REPORTS_DIR, filename)
            with self.subTest(file=filename):
                opinion = self.parser.parse_from_file(path)
                self.assertIn(
                    opinion,
                    MarketReportParser.VALID_OPINIONS,
                    f"{filename}에서 유효한 의견을 추출하지 못함: '{opinion}'",
                )


if __name__ == "__main__":
    print("=" * 70)
    print("MarketReportParser 테스트 시작")
    print("=" * 70)
    unittest.main(verbosity=2)
