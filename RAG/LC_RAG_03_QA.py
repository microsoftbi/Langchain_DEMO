import os
import sys
import logging
from typing import Optional, List, Tuple, Union

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.chat_models import ChatDashScope  # removed: causes ImportError on some versions
import dashscope
from dashscope import Generation
from http import HTTPStatus

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_vectorstore(
    vectorstore_dir: str = "./RAG/chroma_db",
    embedding_model: str = "text-embedding-v4",
    dashscope_api_key: Optional[str] = None,
    collection_name_base: str = "my_collection",
) -> Tuple[Chroma, DashScopeEmbeddings, int, str]:
    """
    与 LC_RAG_01_LoadKnowledgebase.py/LC_RAG_02_RecallTest.py 保持一致：
    - 使用 DashScopeEmbeddings 探测嵌入维度
    - collection 使用 "{base}_dim{emb_dim}"
    - persist_directory 使用模型专属子目录（按模型名派生）
    返回 (vectorstore, embeddings, emb_dim, persist_dir)
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

    vs = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vs, embeddings, emb_dim, persist_dir


def retrieve_context(
    question: str,
    k: int,
    vectorstore: Chroma,
) -> List[str]:
    """使用向量库检索 top-k 文档内容，返回文本片段列表。"""
    docs = vectorstore.similarity_search(question, k=k)
    chunks: List[str] = []
    for d in docs:
        src = d.metadata.get("source", "<unknown>")
        text = d.page_content.strip().replace("\n", " ")
        chunks.append(f"[source: {src}]\n{text}")
    return chunks


def _extract_answer_from_generation_response(resp: Union[dict, object]) -> str:
    """尽量从 DashScope Generation.call 的响应中提取答案文本。"""
    # 字典形式
    if isinstance(resp, dict):
        output = resp.get("output") or {}
        # 优先 text
        text = output.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        # message 格式
        choices = output.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                # content 可能是多段列表，拼接为字符串
                return "".join([str(c) for c in content]).strip()
        # 兜底
        return str(resp)

    # 对象形式（SDK 返回对象），尝试通用属性
    output = getattr(resp, "output", None)
    if output is not None:
        text = getattr(output, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        choices = getattr(output, "choices", None)
        if choices:
            first = choices[0]
            msg = getattr(first, "message", None)
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    return "".join([str(c) for c in content]).strip()
    return str(resp)


def answer_question(
    question: str,
    top_k: int = 5,
    embedding_model: str = "text-embedding-v4",
    chat_model: str = os.getenv("CHAT_MODEL", "qwen-turbo"),
    dashscope_api_key: Optional[str] = None,
    vectorstore_dir: str = "./RAG/chroma_db",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Tuple[str, List[str]]:
    """
    RAG 问答：检索相关上下文，调用 Qwen(DashScope) 生成答案。
    返回 (answer, sources)
    """
    vs, embeddings, emb_dim, persist_dir = build_vectorstore(
        vectorstore_dir=vectorstore_dir,
        embedding_model=embedding_model,
        dashscope_api_key=dashscope_api_key,
    )

    # 检索上下文
    context_chunks = retrieve_context(question, k=top_k, vectorstore=vs)
    sources = []
    for c in context_chunks:
        # 提取 source
        if c.startswith("[source: "):
            end = c.find("]\n")
            if end != -1:
                sources.append(c[len("[source: "):end])
    context_str = "\n\n".join(context_chunks)

    # 构造提示词
    system_prompt = (
        "你是一个严谨的问答助手。请基于提供的检索上下文进行回答，"
        "不要编造信息，若上下文无答案请说明无法从资料中找到。"
    )
    user_prompt = (
        f"问题: {question}\n\n"
        f"检索到的上下文(可能不完整，仅供参考):\n{context_str}\n\n"
        "请给出简洁、准确的中文回答，并在需要时引用关键点。"
    )

    # 使用 DashScope 官方 SDK 生成答案
    if dashscope_api_key is None:
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    dashscope.api_key = dashscope_api_key

    gen_kwargs = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "result_format": "message",  # 优先返回 message 格式，便于解析
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.info("调用 DashScope Generation 生成答案...")
    try:
        resp = Generation.call(**gen_kwargs)
        status_code = getattr(resp, "status_code", None)
        if status_code is None and isinstance(resp, dict):
            status_code = resp.get("status_code")
        if status_code == HTTPStatus.OK or status_code == 200:
            answer = _extract_answer_from_generation_response(resp)
        else:
            msg = getattr(resp, "message", None)
            if msg is None and isinstance(resp, dict):
                msg = resp.get("message")
            logger.error(f"生成失败: status={status_code}, message={msg}")
            answer = "对不起，生成答案时出现错误。"
    except Exception as e:
        logger.error(f"生成答案时出错: {e}")
        answer = "对不起，生成答案时出现错误。"

    return answer.strip(), sources


if __name__ == "__main__":


    question = "哪些节假日应该安排休假？"
    top_k = 5
    embedding_model = "text-embedding-v4"
    chat_model = "qwen-turbo"

    answer, sources = answer_question(
        question=question,
        top_k=top_k,
        embedding_model=embedding_model,
        chat_model=chat_model,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        vectorstore_dir=os.getenv("VECTORSTORE_DIR", "./RAG/chroma_db"),
    )

    print("\n=== Answer ===\n")
    print(answer)
    if sources:
        print("\n=== Sources ===")
        for s in sources:
            print(f"- {s}")