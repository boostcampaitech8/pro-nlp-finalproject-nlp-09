from typing import Dict, Any, Optional, Union, List
import os
import sys
import traceback
import pandas as pd
from google.cloud import bigquery

# 프로젝트 루트를 path에 추가 (app/models -> app -> 프로젝트 루트, dirname 2회)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# daily_prediction_pipeline 내부에서 "from ensemble_predictor import ..." 사용하므로 해당 디렉터리도 path에 추가
_news_sentiment_dir = os.path.join(_project_root, "app", "model", "news_sentiment_model")
if _news_sentiment_dir not in sys.path:
    sys.path.insert(0, _news_sentiment_dir)

# BigQuery 클라이언트
try:
    from app.utils.bigquery_client import BigQueryClient
except ImportError as e:
    print(f"경고: BigQueryClient 임포트 실패: {e}")
    BigQueryClient = None

# daily_prediction_pipeline (BigQuery에서 가져온 데이터로 예측)
try:
    from app.model.news_sentiment_model.daily_prediction_pipeline import run_daily_prediction as _run_daily_prediction
except ImportError:
    try:
        from model.news_sentiment_model.daily_prediction_pipeline import run_daily_prediction as _run_daily_prediction
    except ImportError:
        _run_daily_prediction = None


class SentimentAnalyzer:
    """
    뉴스 감성 분석 및 시장 예측기.
    BigQuery corn_all_news_with_sentiment 테이블에서 데이터를 가져와 daily_prediction_pipeline을 실행합니다.
    """

    def __init__(self):
        pass

    def run_daily_prediction(
        self,
        target_date: str,
        lookback_days: int = 7,
        filter_status: str = "T",
        keyword_filter: Union[str, List[str], None] = "corn and (price or demand or supply or inventory)",
        model_dir: Optional[str] = None,
        output_dir: str = "outputs",
        save_file: bool = False,
    ) -> Dict[str, Any]:
        """
        target_date, lookback_days 기준으로 BigQuery에서 뉴스/가격을 불러와 일일 예측을 수행합니다.
        filter_status는 'T' 고정, keyword는 기본 'corn and (price or demand or supply or inventory)'.

        Args:
            target_date: 예측 기준 날짜 (YYYY-MM-DD)
            lookback_days: 뉴스 lookback 일수
            filter_status: filter_status 필터 (기본 'T' 고정)
            keyword_filter: key_word 필터 (기본: corn and (price or demand or supply or inventory))
            model_dir: 앙상블 모델 디렉토리 (None이면 news_sentiment_model/trained_models_noPrice 사용)
            output_dir: 보고서 저장 디렉토리
            save_file: True면 예측 결과 JSON 파일 저장

        Returns:
            Dict: 예측 보고서 (metadata, prediction, evidence, market_analysis 등). 실패 시 error 키 포함.
        """
        if not BigQueryClient:
            return {"error": "BigQueryClient를 사용할 수 없습니다."}
        if not _run_daily_prediction:
            return {"error": "daily_prediction_pipeline을 임포트할 수 없습니다."}

        try:
            if model_dir is None:
                _dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(_dir, "..", "model", "news_sentiment_model", "trained_models_noPrice")
                model_dir = os.path.normpath(model_dir)
            bq = BigQueryClient()
            news_df = bq.get_news_for_prediction(
                target_date,
                lookback_days=lookback_days,
                filter_status=filter_status,
                keyword_filter=keyword_filter,
            )
            price_df = bq.get_price_history(target_date, lookback_days=30)

            if news_df.empty:
                return {"error": f"{target_date} 기준 최근 뉴스 데이터가 없습니다."}
            if price_df.empty:
                return {"error": f"{target_date} 기준 최근 가격 데이터가 없습니다."}

            report = _run_daily_prediction(
                target_date=target_date,
                lookback_days=lookback_days,
                news_df=news_df,
                price_df=price_df,
                model_dir=model_dir,
                output_dir=output_dir,
                save_file=save_file,
            )
            if report is None:
                return {"error": f"{target_date} 예측 데이터 준비 실패 (뉴스/임베딩 부족)."}
            # llm_summarizer 호환: evidence_news = supporting (최대 3개) + opposing (최대 3개)
            evidence = report.get("evidence") or {}
            supporting = evidence.get("supporting_news") or []
            opposing = evidence.get("opposing_news") or []
            evidence_news = [
                {
                    "title": x.get("title", ""),
                    # all_text 우선 사용 (없으면 description 폴백)
                    "all_text": x.get("all_text") or x.get("description", ""),
                    "impact_score": x.get("impact_score"),
                    "sentiment": x.get("sentiment", ""),
                }
                for x in (supporting[:3] + opposing[:3])
            ]
            report["evidence_news"] = evidence_news
            
            # sentiment_summary를 BigQuery에 저장
            market_analysis = report.get("market_analysis") or {}
            sentiment_summary = market_analysis.get("sentiment_summary") or {}
            if sentiment_summary:
                try:
                    news_count_val = int(sentiment_summary.get("total_news_count", 0))
                    pos_ratio_val = float(sentiment_summary.get("positive_ratio", 0.0))
                    neg_ratio_val = float(sentiment_summary.get("negative_ratio", 0.0))
                    print(f"[DEBUG] sentiment_summary 저장 시도: date={target_date}, news_count={news_count_val}, pos_ratio={pos_ratio_val}, neg_ratio={neg_ratio_val}")
                    success = self._save_sentiment_result_to_bq(
                        target_date=target_date,
                        news_count=news_count_val,
                        pos_ratio=pos_ratio_val,
                        neg_ratio=neg_ratio_val,
                        keyword="corn",
                    )
                    if success:
                        print(f"✓ sentiment_result_pos_neg 테이블 저장 성공")
                    else:
                        print(f"✗ sentiment_result_pos_neg 테이블 저장 실패")
                except Exception as e:
                    print(f"경고: sentiment_result_pos_neg 테이블 저장 중 예외 발생: {e}")
                    traceback.print_exc()
            
            return report
        except Exception as e:
            traceback.print_exc()
            return {"error": f"시장 예측 중 오류 발생: {str(e)}"}

    def _save_sentiment_result_to_bq(
        self,
        target_date: str,
        news_count: int,
        pos_ratio: float,
        neg_ratio: float,
        keyword: str = "corn",
        dataset_id: Optional[str] = None,
        table_id: str = "sentiment_result_pos_neg",
    ) -> bool:
        """
        sentiment_result_pos_neg 테이블에 감성 분석 결과를 삽입합니다.

        Args:
            target_date: 분석 날짜 (YYYY-MM-DD)
            news_count: 총 뉴스 개수
            pos_ratio: 긍정 비율 (0~1)
            neg_ratio: 부정 비율 (0~1)
            keyword: 키워드 (기본: "corn")
            dataset_id: 데이터셋 ID (None이면 환경변수 사용)
            table_id: 테이블 ID (기본: sentiment_result_pos_neg)

        Returns:
            bool: 성공 여부
        """
        try:
            from datetime import datetime
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            print(f"경고: 잘못된 날짜 형식: {target_date}")
            return False

        dataset = dataset_id or os.getenv("BIGQUERY_DATASET_ID", "tilda")
        project_id = os.getenv("BIGQUERY_PROJECT_ID") or os.getenv("VERTEX_AI_PROJECT_ID")
        if project_id:
            client = bigquery.Client(project=project_id)
        else:
            client = bigquery.Client()
        
        full_table = f"{client.project}.{dataset}.{table_id}"

        try:
            # 기존 데이터 삭제 (중복 방지)
            delete_query = f"DELETE FROM `{full_table}` WHERE date = @date AND keyword = @keyword"
            client.query(
                delete_query,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("date", "DATE", target_date),
                        bigquery.ScalarQueryParameter("keyword", "STRING", keyword),
                    ]
                ),
            ).result()

            # 새 데이터 삽입
            df = pd.DataFrame([{
                "date": target_date,
                "news_count": int(news_count),
                "pos_ratio": float(pos_ratio),
                "neg_ratio": float(neg_ratio),
                "keyword": keyword,
            }])
            # date 컬럼을 DATE 타입으로 변환
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )
            client.load_table_from_dataframe(df, full_table, job_config=job_config).result()
            return True
        except Exception as e:
            print(f"경고: sentiment_result_pos_neg 테이블 저장 실패: {e}")
            traceback.print_exc()
            return False

    def predict_market_impact(self, target_date: str, lookback_days: int = 7) -> Dict[str, Any]:
        """
        특정 날짜의 뉴스 기반 시장 예측 (llm_summarizer 등 기존 호환용).
        BigQuery에서 데이터를 불러와 daily_prediction_pipeline으로 예측합니다.

        Args:
            target_date: 분석할 날짜 (YYYY-MM-DD)
            lookback_days: 뉴스 lookback 일수 (기본 7)

        Returns:
            Dict: 예측 보고서 또는 error 메시지
        """
        return self.run_daily_prediction(
            target_date=target_date,
            lookback_days=lookback_days,
            filter_status="T",
            save_file=False,
        )


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print("\n시장 예측 테스트 (target_date, lookback_days=7):")
    print(analyzer.predict_market_impact("2025-11-13", lookback_days=7))
