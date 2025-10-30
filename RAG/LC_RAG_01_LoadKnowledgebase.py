import os
import logging
from typing import Optional

from langchain_community.document_loaders import TextLoader, DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
import requests
from dotenv import load_dotenv

"""
加载知识库到 Chroma 向量数据库。
"""



load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_documents_to_vectorstore(
    document_dir: str = "./RAG/knowledge_base",
    vectorstore_dir: str = "./RAG/chroma_db",
    embedding_model: str = "text-embedding-v1",
    dashscope_api_key: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str = "my_collection",
) -> bool:
    """
    读取知识库目录，生成文本块，写入到 Chroma 向量数据库。

    - 为避免嵌入维度冲突（不同模型维度不同），将根据嵌入维度自动切换到模型专属子目录并在集合名追加维度后缀。
    参数:
    - document_dir: 文档目录路径
    - vectorstore_dir: 向量数据库持久化目录
    - embedding_model: DashScope 的嵌入模型名称，默认 text-embedding-v1
    - dashscope_api_key: DashScope API Key（默认从环境变量 DASHSCOPE_API_KEY 读取）
    - chunk_size: 文本块大小
    - chunk_overlap: 文本块重叠
    - collection_name: 集合名称

    返回:
    - True: 成功
    - False: 失败
    """

    try:
        # 文档目录检查
        if not os.path.exists(document_dir):
            logger.error(f"文档目录不存在: {document_dir}")
            return False

        logger.info(f"开始从 {document_dir} 加载文档...")

        documents = []

        # 加载 txt
        txt_loader = DirectoryLoader(document_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())

        # 加载 docx
        docx_loader = DirectoryLoader(document_dir, glob="**/*.docx", loader_cls=Docx2txtLoader)
        documents.extend(docx_loader.load())

        if not documents:
            logger.warning("没有加载到任何文档")
            return False

        logger.info(f"成功加载 {len(documents)} 个文档")

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"文档分割完成，生成 {len(splits)} 个文本块")

        # 嵌入模型初始化（Qwen/DashScope）
        if dashscope_api_key is None:
            dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        if not dashscope_api_key:
            logger.error("未提供 DashScope API Key。请设置环境变量 DASHSCOPE_API_KEY 或传入参数 dashscope_api_key。")
            return False

        embeddings = DashScopeEmbeddings(model=embedding_model, dashscope_api_key=dashscope_api_key)
        logger.info("开始生成嵌入并写入向量库，这个过程可能较慢，取决于文本块数量与模型大小...")

        # 探测嵌入维度，并调整集合名与持久化目录，避免与旧集合维度冲突
        try:
            probe_vec = embeddings.embed_query("dimension probe")
            emb_dim = len(probe_vec)
            logger.info(f"嵌入维度: {emb_dim}")
            # 在集合名追加维度后缀
            collection_name = f"{collection_name}_dim{emb_dim}"
            logger.info(f"使用集合名称: {collection_name}")
        except Exception as e:
            logger.warning(f"无法探测嵌入维度，将使用原集合名，可能出现维度不匹配: {e}")
            emb_dim = None

        # 针对不同模型使用独立子目录，避免与历史集合冲突
        model_dir_tag = embedding_model.replace(":", "_").replace("/", "_")
        persist_dir = os.path.join(vectorstore_dir, model_dir_tag)

        # 确保向量库目录存在
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"向量数据库将持久化到: {persist_dir}")

        # 创建并持久化 Chroma（使用模型专属目录与维度后缀集合名）
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

        # 触发持久化到磁盘
        vectorstore.persist()
        logger.info("向量数据库已创建并持久化，正在验证...")

        # 再次打开进行验证
        test_vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        test_docs = test_vectorstore.get()
        test_count = len(test_docs.get("ids", []))
        logger.info(f"验证完成: 找到 {test_count} 条向量记录")

        # 列出持久化目录内容
        logger.info("持久化目录内容:")
        for f in os.listdir(persist_dir):
            p = os.path.join(persist_dir, f)
            try:
                size_kb = os.path.getsize(p) / 1024
                logger.info(f"- {f} ({size_kb:.2f} KB)")
            except Exception:
                logger.info(f"- {f}")

        logger.info("文档加载到向量数据库成功")
        return True

    except Exception as e:
        logger.exception(f"加载文档到向量数据库时出错: {e}")
        return False


if __name__ == "__main__":
    success = load_documents_to_vectorstore(
        document_dir="./RAG/knowledge_base",
        vectorstore_dir="./RAG/chroma_db",
        embedding_model="text-embedding-v4",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    print("文档已成功加载到向量数据库" if success else "文档加载失败")