import json

path = "final_processed_news.json"
with open(path, "r", encoding="utf-8") as f:
    items = json.load(f)

filtered = [it for it in items if it.get("filter_status") == "T"]

with open(path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=4)

print(len(filtered))
