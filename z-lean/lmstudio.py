import asyncio
from dotenv import load_dotenv
import os
from functools import partial
from lightrag import LightRAG
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import json
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

RAG_STORAGE = os.environ.get(
    "RAG_STORAGE", "C:/Users/gqly/Desktop/workspace/RAG-Anything/z-lean/rag_storage"
)

LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-4-e2b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "lmstudio")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1234/v1")

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5-embedding"
)
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "lmstudio")
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:1234/v1")

VISION_MODEL = os.environ.get("VISION_MODEL", "google/gemma-4-e2b")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "lmstudio")
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "http://127.0.0.1:1234/v1")


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
            api_key=VISION_API_KEY,
            base_url=VISION_BASE_URL,
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
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
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
            print(f"Testing LM Studio connection at: {LLM_BASE_URL}")
            client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
            models = await client.models.list()
            print(f"Connected successfully! Found {len(models.data)} models")

            # Show available models
            print("Available models:")
            for i, model in enumerate(models.data[:5]):
                marker = "✅" if model.id == LLM_MODEL else "  "
                print(f"{marker} {i+1}. {model.id}")

            if len(models.data) > 5:
                print(f"  ... and {len(models.data) - 5} more models")

            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            print("\nTroubleshooting tips:")
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
            working_dir=RAG_STORAGE,
            llm_model_func=llm_model_func_factory,
            embedding_func=embedding_func_factory(),
            default_llm_timeout=1800,
            default_embedding_timeout=1800,
        )
        config = RAGAnythingConfig(
            parser="mineru-open-api",
        )
        await self.light_rag.initialize_storages()
        self.rag = RAGAnything(
            lightrag=self.light_rag,
            vision_model_func=vision_model_func_factory,
            config=config,
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
        """

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
                try:
                    with open(content_list_json_path, "r", encoding="utf-8") as f:
                        content_list = json.load(f)
                        # 将 img_path 转换为绝对路径
                        for content in content_list:
                            if content["type"] == "image":
                                img_path = content["img_path"]
                                abs_img_path = os.path.join(root, img_path)
                                abs_img_path = os.path.abspath(abs_img_path)
                                content["img_path"] = abs_img_path
                except Exception as e:
                    print(f"Error reading {content_list_json_path}: {str(e)}")
                    continue
                # 分批插入知识库，每个批次100条记录
                batch_size = 100
                # 读取断点，如果存在则从断点继续插入
                with open(
                    f"{file_name}.last_inserted_index.txt", "w+", encoding="utf-8"
                ) as f:
                    last_inserted_index = (
                        int(f.read().strip()) if f.read().strip() else 0
                    )
                for i in tqdm(
                    range(last_inserted_index, len(content_list), batch_size),
                    desc=f"Inserting {file_name} into RAG",
                ):
                    # 不能超过列表长度
                    end_index = min(i + batch_size, len(content_list))
                    batch = content_list[i:end_index]
                    await self.rag.insert_content_list(
                        content_list=batch,
                        file_path=file_name,  # 用于引用的参考文件名
                        split_by_character=split_by_character,  # 可选的文本分割
                        split_by_character_only=split_by_character_only,  # 可选的文本分割模式
                        doc_id=doc_id,  # 可选的自定义文档ID（如果未提供将自动生成）
                        display_stats=display_stats,  # 显示内容统计信息
                    )
                    # 写入当前插入的索引
                    with open(
                        f"{file_name}.last_inserted_index.txt", "w", encoding="utf-8"
                    ) as f:
                        f.write(str(end_index))


async def main():
    my_rag = MyRagAnything()
    await my_rag.initialize_rag()
    # await my_rag.test_connection()
    # await my_rag.rag.process_document_complete(
    #     file_path=r"金融时间序列分析 第3版.pdf",
    #     output_dir="./output_lmstudio",
    #     display_stats=True,
    # )
    # for file_name in os.listdir("output_lmstudio"):
    #     print("="*50)
    #     print(file_name)
    #     print("="*50)
    #     await my_rag.insert_mineru_content(
    #         file_name=file_name,
    #         file_root=os.path.join("output_lmstudio", file_name),
    #         doc_id=file_name,
    #         display_stats=True,
    #     )
    query_result = await my_rag.rag.aquery(
        "给我几个量化分析的方案,同时给出对应的公式或流程",
        system_prompt="必须使用中文回答，公式必须有引用",
    )
    print("=" * 50)
    print(query_result)
    print("=" * 50)
    pass


if __name__ == "__main__":
    asyncio.run(main())
