import datasets
import json

print("正在加载 iwslt2017 (en-de) 数据集 (从本地缓存)...")

# --- (核心修改) ---
# 
# 
# 
try:
    dataset = datasets.load_dataset("iwslt2017", "iwslt2017-en-de")
except Exception as e:
    print(f"加载失败。请确保你本地的缓存是完好的。")
    print(f"错误: {e}")
    exit()
# --- (修改结束) ---

print("数据集加载成功！")

# 2. 
# 

def save_to_jsonl(split_name, data):
    filename = f"iwslt2017_{split_name}.jsonl"
    count = 0
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            # 
            f.write(json.dumps(item) + "\n")
            count += 1
    print(f"已成功导出 {count} 条样本到 {filename}")

save_to_jsonl("train", dataset["train"])
save_to_jsonl("validation", dataset["validation"])

print("\n数据导出完成！你现在可以把 'iwslt2017_train.jsonl' 和 'iwslt2017_valid.jsonl' 上传到服务器了。")