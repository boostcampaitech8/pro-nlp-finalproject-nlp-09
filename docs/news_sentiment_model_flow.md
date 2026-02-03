run_pipeline.py
    │
    └─► orchestrate_analysis() [app/routes/orchestrator.py]
            │
            └─► LLMSummarizer.summarize() [app/models/llm_summarizer.py]
                    │
                    └─► LangChain Agent가 Tool 호출
                            │
                            ├─► timeseries_predictor (시계열 예측)
                            │
                            └─► news_sentiment_analyzer [app/models/sentiment_analyzer.py]
                                    │
                                    └─► SentimentAnalyzer.predict_market_impact()
                                            │
                                            ├─► BigQueryClient.get_news_for_prediction() ⚠️ 레거시
                                            ├─► BigQueryClient.get_price_history() ⚠️ 레거시
                                            │
                                            ├─► preprocess_news_data() [preprocessing.py]
                                            │
                                            └─► CornPricePredictor.predict_with_evidence()
                                                    [inference_with_evidence.py]
                                                    │
                                                    ├─► prepare_inference_features() [preprocessing.py]
                                                    └─► extract_evidence_news()