import os
import json

root = r"C:\Users\gqly\Desktop\workspace\RAG-Anything\z-lean\output_lmstudio"
# 遍历 output_lmstudio下的文件夹
for folder in os.listdir(root):
    # 读取文件夹内的json文件
    os.path.join(root, folder)
    with open(os.path.join(root, folder, f"{folder}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
        for i in data:
            if i.get("img_path"):
                i["img_path"] = os.path.join(folder, i["img_path"])
                print(i["img_path"])

