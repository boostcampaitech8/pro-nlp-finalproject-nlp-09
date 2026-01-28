import boto3
import json
import logging
import os

class TitanEmbedder:
    def __init__(self, region_name="us-east-1"):
        # Airflow Variables에서 AWS credentials 가져오기
        try:
            from airflow.models import Variable
            aws_access_key = Variable.get("AWS_ACCESS_KEY_ID", default_var=None)
            aws_secret_key = Variable.get("AWS_SECRET_ACCESS_KEY", default_var=None)
            aws_region = Variable.get("AWS_DEFAULT_REGION", default_var=region_name)
            
            # 환경변수 설정 (boto3가 사용할 수 있도록)
            if aws_access_key:
                os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
            if aws_secret_key:
                os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
            if aws_region:
                region_name = aws_region
        except Exception as e:
            logging.warning(f"Airflow Variables에서 AWS credentials을 못 읽었습니다: {e}. 환경변수 사용")
        
        # AWS Bedrock 접속 설정
        self.bedrock = boto3.client(
            service_name='bedrock-runtime', 
            region_name=region_name
        )
        self.model_id = "amazon.titan-embed-text-v2:0"

    def generate_embedding(self, text, dimensions=512):
        """텍스트를 받아 Titan v2 임베딩 벡터를 반환합니다."""
        try:
            body = json.dumps({
                "inputText": text,
                "dimensions": dimensions, # 256, 512, 1024 중 선택
                "normalize": True
            })
            
            response = self.bedrock.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('embedding')
            
        except Exception as e:
            logging.error(f" 임베딩 생성 실패: {e}")
            return None