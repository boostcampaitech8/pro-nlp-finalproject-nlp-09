NEWS_API_KEY = "bd8ffa6d3c5f48cd8f66b18778ff305b"
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