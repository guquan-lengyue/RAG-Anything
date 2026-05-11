import os
import subprocess
from pathlib import Path
from pypdf import PdfReader

MINERU_TOKEN = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MDMwNjUwNSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3ODExNjM4MiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiM2VmOTI0YjEtM2NjNS00MzgyLWI0MjktNDEzY2NmZjYxZGU4IiwiZW1haWwiOiIiLCJleHAiOjE3ODU4OTIzODJ9.3yGG9tq-Cag45Q83TrwZxPBelBXm7ilGEY_SStG7hIdQ4pQezwP5aPQsqqdQX33hwmCei4GaWxM8GsIiNcbJZA"
os.environ["MINERU_TOKEN"] = MINERU_TOKEN
# PDF目录
PDF_DIR = r"C:\Users\gqly\Desktop\finance"

# mineru 命令路径
MINERU_EXE = r"mineru-open-api"

# 每次处理页数
CHUNK_SIZE = 200


def get_pdf_page_count(pdf_path: str) -> int:
    """获取PDF页数"""
    reader = PdfReader(pdf_path)
    return len(reader.pages)


def run_mineru(pdf_path: str, start_page: int, end_page: int):
    """执行 mineru 命令"""

    pdf_name = Path(pdf_path).stem

    # 输出目录
    output_dir = os.path.join(
        os.path.dirname(pdf_path), f"{pdf_name}_{start_page}_{end_page}"
    )

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        MINERU_EXE,
        "extract",
        pdf_path,
        "-f",
        "md,json",
        "--ocr",
        "--model",
        "vlm",
        "--pages",
        f"{start_page}-{end_page}",
        "-o",
        output_dir,
        "--formula",
        "--timeout",
        "1200",
    ]

    print("=" * 80)
    print("执行命令:")
    print(" ".join(cmd))
    print("=" * 80)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False,
        )
        print(result.stdout)

        print(f"完成: {pdf_name} [{start_page}-{end_page}]")

    except subprocess.CalledProcessError as e:
        print(f"执行失败: {pdf_name} [{start_page}-{end_page}]")
        print(e)


def process_pdf(pdf_path: str):
    """处理单个PDF"""
    total_pages = get_pdf_page_count(pdf_path)
    print(f"\nPDF: {pdf_path}")
    print(f"总页数: {total_pages}")

    start = 1

    while start <= total_pages:
        end = min(start + CHUNK_SIZE - 1, total_pages)
        run_mineru(pdf_path, start, end)
        start += CHUNK_SIZE


def main():
    pdf_files = []
    for root, dirs, files in os.walk(PDF_DIR):
        for file in files:
            if file.lower().endswith(".pdf") and "金融时间序列分析-第3版" in file:
                pdf_files.append(os.path.join(root, file))
        
    print(f"发现 {len(pdf_files)} 个PDF文件")

    for pdf_file in pdf_files:
        try:
            process_pdf(pdf_file)
        except Exception as e:
            print(f"处理失败: {pdf_file}")
            print(e)


if __name__ == "__main__":
    main()
