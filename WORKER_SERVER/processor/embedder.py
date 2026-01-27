import boto3
import json
import logging

class TitanEmbedder:
    def __init__(self, region_name="us-east-1"):
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