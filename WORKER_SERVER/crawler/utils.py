import hashlib
from datetime import datetime

def generate_md5_id(url):
    "doc_url을 기반으로 고유 ID 생성"
    return hashlib.md5(url.encode()).hexdigest()


def format_date(iso_date):
    "News API 형식으로 YYYY-MM-DD HH:MM:SS 변환"
    if not iso_date:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
    return dt.strftime('%Y-%m-%d %H:%M:%S')


