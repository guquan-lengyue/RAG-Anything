import asyncio
from dotenv import load_dotenv
import os
from functools import partial
from lightrag import LightRAG
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, setup_logger
from raganything.modalprocessors import ImageModalProcessor, TableModalProcessor
import json
from typing import List, Dict, Optional
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1234")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "lmstudio")
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-4-e2b")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5-embedding"
)
VISION_MODEL = os.environ.get("VISION_MODEL", "google/gemma-4-e2b")


async def llm_model_func_factory(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs,
):
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def vision_model_func_factory(
    prompt,
    system_prompt=None,
    history_messages=[],
    image_data=None,
    messages=None,
    **kwargs,
):
    if image_data is not None:
        return await openai_complete_if_cache(
            VISION_MODEL,
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                (
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None
                ),
                (
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt}
                ),
            ],
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            **kwargs,
        )
    else:
        return await llm_model_func_factory(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            messages=messages,
            **kwargs,
        )

def embedding_func_factory():
    return EmbeddingFunc(
        embedding_dim=768,
        max_token_size=2048,
        func=partial(
            openai_embed.func,
            model=EMBEDDING_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        ),
    )


class MyRagAnything:
    def __init__(self):
        self.light_rag = None
        self.rag = None
        pass

    async def test_connection(self) -> bool:
        """Test LM Studio connection."""
        try:
            print(f"🔌 Testing LM Studio connection at: {LLM_BASE_URL}")
            client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
            models = await client.models.list()
            print(f"✅ Connected successfully! Found {len(models.data)} models")

            # Show available models
            print("📊 Available models:")
            for i, model in enumerate(models.data[:5]):
                marker = "🎯" if model.id == LLM_MODEL else "  "
                print(f"{marker} {i+1}. {model.id}")

            if len(models.data) > 5:
                print(f"  ... and {len(models.data) - 5} more models")

            return True
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            print("\n💡 Troubleshooting tips:")
            print("1. Ensure LM Studio is running")
            print("2. Start the local server in LM Studio")
            print("3. Load a model or enable just-in-time loading")
            print(f"4. Verify server address: {LLM_BASE_URL}")
            return False
        finally:
            try:
                await client.close()
            except Exception:
                pass



    async def initialize_rag(self):
        self.light_rag = LightRAG(
            working_dir="./rag_storage",
            llm_model_func=llm_model_func_factory,
            embedding_func=embedding_func_factory(),
        )
        await self.light_rag.initialize_storages()
        self.rag = RAGAnything(
            lightrag=self.light_rag,
            vision_model_func=vision_model_func_factory,
        )

    async def insert_mineru_content(
        self,
        file_name,
        file_root,
        split_by_character=None,
        split_by_character_only=False,
        doc_id=None,
        display_stats=True,
    ):
        """_summary_

        Args:
            file_name (str): 用于引用的参考文件名
            file_root (str): 文件根目录
            split_by_character (str, optional): 可选的文本分割字符. Defaults to None.
            split_by_character_only (bool, optional): 可选的文本分割模式. Defaults to False.
            doc_id (str, optional): 可选的自定义文档ID（如果未提供将自动生成）. Defaults to None.
            display_stats (bool, optional): 显示内容统计信息. Defaults to True.
        """
        # 查找file_root目录下的json文件
        for root, dirs, files in os.walk(file_root):
            for file in files:
                if not file.endswith(".json"):
                    continue
                content_list_json_path = os.path.join(root, file)
                # 读取json文件内容
                with open(content_list_json_path, "r", encoding="utf-8") as f:
                    content_list = json.load(f)
                    # 将 img_path 转换为绝对路径
                    for content in content_list:
                        if content["type"] == "image":
                            img_path = content["img_path"]
                            abs_img_path = os.path.join(root, img_path)
                            content["img_path"] = abs_img_path
                    # 插入知识库
                    await self.rag.insert_content_list(
                        content_list=content_list,
                        file_path=file_name,  # 用于引用的参考文件名
                        split_by_character=split_by_character,  # 可选的文本分割
                        split_by_character_only=split_by_character_only,  # 可选的文本分割模式
                        doc_id=doc_id,  # 可选的自定义文档ID（如果未提供将自动生成）
                        display_stats=display_stats,  # 显示内容统计信息
                    )


async def main():
    my_rag = MyRagAnything()
    await my_rag.test_connection()
    await my_rag.initialize_rag()
    await my_rag.insert_mineru_content(
        file_name="利用Python进行数据分析.pdf",
        file_root=r"C:\Users\kang_\Desktop\my\rag-anything\z-lean\output_lmstudio\利用Python进行数据分析_c00bc2b7",
    )
    # results = await my_rag.rag.query(
    #     "请基于利用Python进行数据分析.pdf的内容，介绍一下什么是数据清洗？", mode="vlm"
    # )
    # print(results)
    pass


if __name__ == "__main__":
    asyncio.run(main())
