import os
import sys
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_vectorstore(
    vectorstore_dir: str = "./RAG/chroma_db",
    embedding_model: str = "text-embedding-v4",
    dashscope_api_key: Optional[str] = None,
    collection_name_base: str = "my_collection",
) -> Chroma:
    """
    根据 LC_RAG_01_LoadKnowledgebase.py 的持久化策略，加载 Chroma 向量库：
    - 使用 DashScopeEmbeddings 计算嵌入维度
    - collection 名使用 "{base}_dim{emb_dim}"
    - persist_directory 使用模型专属子目录（按模型名派生）
    """
    if dashscope_api_key is None:
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise RuntimeError("缺少 DashScope API Key，请设置环境变量 DASHSCOPE_API_KEY 或传入参数。")

    embeddings = DashScopeEmbeddings(model=embedding_model, dashscope_api_key=dashscope_api_key)

    # 探测嵌入维度与持久化目录
    probe_vec = embeddings.embed_query("dimension probe")
    emb_dim = len(probe_vec)
    collection_name = f"{collection_name_base}_dim{emb_dim}"
    model_dir_tag = embedding_model.replace(":", "_").replace("/", "_")
    persist_dir = os.path.join(vectorstore_dir, model_dir_tag)

    logger.info(f"加载向量库: dir={persist_dir}, collection={collection_name}, dim={emb_dim}")

    # 以相同的 embedding_function 打开向量库
    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vs


def recall(
    query: str,
    top_k: int = 5,
    vectorstore_dir: str = "./RAG/chroma_db",
    embedding_model: str = "text-embedding-v4",
    dashscope_api_key: Optional[str] = None,
) -> None:
    """
    召回测试：对 query 执行相似度检索并打印结果。
    """
    vs = build_vectorstore(
        vectorstore_dir=vectorstore_dir,
        embedding_model=embedding_model,
        dashscope_api_key=dashscope_api_key,
    )

    logger.info(f"执行相似度检索: k={top_k}, query='{query}'")
    docs = vs.similarity_search(query, k=top_k)

    print("\n=== Recall Results ===")
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "<unknown>")
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        print(f"[{i}] source={src}\n    {snippet}\n")


if __name__ == "__main__":
    

    #query = "哪些节假日应该安排休假?"
    query = "什么是未成年?"
    top_k = 5
    embedding_model = "text-embedding-v4"

    recall(
        query=query,
        top_k=top_k,
        vectorstore_dir=os.getenv("VECTORSTORE_DIR", "./RAG/chroma_db"),
        embedding_model=embedding_model,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )