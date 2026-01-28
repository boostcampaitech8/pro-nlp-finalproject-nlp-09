NEWS_API_KEY = "bd8ffa6d3c5f48cd8f66b18778ff305b"
CURRENTS_API_KEY ="-Sm28Fd_uH6-I9D5HYn50fKpER2GTL996C4WukmXKq0hWXnB"
TARGET_QUERIES = [
    "corn AND (price OR demand OR supply OR inventory)",
    "soybean AND (price OR demand OR supply OR inventory)",
    "wheat AND (price OR demand OR supply OR inventory)",
    '"United States Department of Agriculture" OR USDA'
]

COMMODITY_MAP = {
    "corn": "corn",
    "soybean": "soybean",
    "wheat": "wheat",
    "usda": "all"
}