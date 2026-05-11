import os
import shutil
import json

src_dir = r"C:\Users\gqly\Desktop\finance\out"
out_dir = "./out"
file_names = os.listdir(src_dir)



current = ""
# 将相同书名的文件夹进行内容合并
# 1. images 目录内容合并
# 2. json 文件内容合并
# 3. md 文件内容合并
for src_file_name in file_names:
    if "_" not in src_file_name:
        continue
    book_name = src_file_name[:src_file_name.find("_")]
    if book_name != current:
        print(f"合并 {book_name} 的内容")
        # 创建合并后的文件夹
        os.makedirs(os.path.join(out_dir, book_name), exist_ok=True)
        # 创建合并后的image 目录
        os.makedirs(os.path.join(out_dir, book_name, "images"), exist_ok=True)
        current = book_name
    src_file_path = os.path.join(src_dir, src_file_name)
    src_image_dir = os.path.join(src_file_path, "images")
    src_json_file = os.path.join(src_file_path, f"{book_name}.json")
    src_md_file = os.path.join(src_file_path, f"{book_name}.md")

    target_file_path = os.path.join(out_dir, book_name)
    target_image_dir = os.path.join(target_file_path, "images")
    target_json_file = os.path.join(target_file_path, f"{book_name}.json")
    target_md_file = os.path.join(target_file_path, f"{book_name}.md")

    # 合并 images 目录内容
    # 如果images 目录不存在则跳过
    if os.path.exists(src_image_dir):
        for image_file in os.listdir(src_image_dir):
            image_path = os.path.join(src_image_dir, image_file)
            # 复制文件到合并后的目录
            shutil.copy(image_path, os.path.join(target_image_dir, image_file))
    # 以文件追加写入的方式合并markdown文件,文件不存在则创建新文件
    with open(target_md_file, "a", encoding="utf-8") as f:
        with open(src_md_file, "r", encoding="utf-8") as md:
            f.write(md.read())
    # 以list.extend的方式合并json文件
    with open(target_json_file, "w+", encoding="utf-8") as target_json_f:
        with open(src_json_file, "r", encoding="utf-8") as src_json_f:
            try:
                target_json = json.load(target_json_f)
            except json.JSONDecodeError:
                target_json = []
            src_json = json.load(src_json_f)
            target_json.extend(src_json)
            json.dump(target_json, target_json_f, ensure_ascii=False, indent=4)