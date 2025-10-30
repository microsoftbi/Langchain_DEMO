import os
import sys
import json
import logging
from typing import Dict, List, Any
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
from http import HTTPStatus
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

def _build_client(persist_path: str):
    """构造 Chroma 客户端，兼容不同版本 chromadb。"""
    try:
        from chromadb import PersistentClient
        return PersistentClient(path=persist_path)
    except Exception:
        try:
            from chromadb import Client
            from chromadb.config import Settings
            return Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_path))
        except Exception as e:
            raise RuntimeError(f"无法初始化 Chroma 客户端: {e}")


def _safe_count(collection) -> int:
    try:
        return int(collection.count())
    except Exception:
        return -1


def _persist_paths(vectorstore_dir: str) -> List[str]:
    """根据根目录枚举可能的持久化路径（模型子目录或根目录本身）。"""
    if not os.path.exists(vectorstore_dir):
        return []
    subdirs = [
        os.path.join(vectorstore_dir, d)
        for d in os.listdir(vectorstore_dir)
        if os.path.isdir(os.path.join(vectorstore_dir, d))
    ]
    return subdirs if subdirs else [vectorstore_dir]


def list_chroma_collections(vectorstore_dir: str = "./RAG/chroma_db") -> Dict[str, List[Dict[str, Any]]]:
    """
    扫描指定向量库根目录，列出各模型子目录下的所有 Chroma collections。
    返回结构: { persist_path: [ {name, metadata, count}, ... ], ... }
    """
    results: Dict[str, List[Dict[str, Any]]] = {}

    paths = _persist_paths(vectorstore_dir)
    if not paths:
        logger.warning(f"路径不存在: {vectorstore_dir}")
        return results

    for p in paths:
        try:
            client = _build_client(p)
            collections = client.list_collections() or []
            infos = []
            for c in collections:
                info = {
                    "name": getattr(c, "name", None),
                    "metadata": getattr(c, "metadata", {}) or {},
                    "count": _safe_count(c),
                }
                infos.append(info)
            results[p] = infos
            logger.info(f"列出集合: {p} -> {len(infos)} 个")
        except Exception as e:
            logger.error(f"读取 {p} 失败: {e}")
            results[p] = [{"error": str(e)}]

    return results


def _decode_unicode_escapes(text: str) -> str:
    try:
        return codecs.decode(text, 'unicode_escape')
    except Exception:
        try:
            return bytes(text, 'utf-8').decode('unicode_escape')
        except Exception:
            return text


def list_collection_sources(vectorstore_dir: str, collection_name: str) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    paths = _persist_paths(vectorstore_dir)
    if not paths:
        return results
    for p in paths:
        try:
            client = _build_client(p)
            try:
                coll = client.get_collection(name=collection_name)
            except Exception:
                # 在此持久化路径下未找到该集合
                continue
            data = coll.get(include=["metadatas"]) or {}
            metas = data.get("metadatas") or []
            counter: Dict[str, int] = {}
            for meta in metas:
                src = None
                if isinstance(meta, dict):
                    src = meta.get("source") or meta.get("path")
                if isinstance(src, str):
                    src = _decode_unicode_escapes(src)
                else:
                    src = None
                if not src:
                    src = "(unknown)"
                counter[src] = counter.get(src, 0) + 1
            items = [{"source": s, "count": c} for s, c in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]
            results[p] = items
            logger.info(f"集合来源: {p}/{collection_name} -> {len(items)} 个来源")
        except Exception as e:
            logger.error(f"读取来源 {p}/{collection_name} 失败: {e}")
            results[p] = [{"error": str(e)}]
    return results


def get_collection_entities(vectorstore_dir: str, collection_name: str, raw: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    paths = _persist_paths(vectorstore_dir)
    if not paths:
        return results

    for p in paths:
        try:
            client = _build_client(p)
            try:
                coll = client.get_collection(name=collection_name)
            except Exception:
                # 在此持久化路径下未找到该集合
                continue
            data = coll.get(include=["metadatas", "documents"]) or {}
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            items: List[Dict[str, Any]] = []
            for i, id_ in enumerate(ids):
                meta = metas[i] if i < len(metas) else {}
                doc_val = docs[i] if i < len(docs) else None
                if isinstance(doc_val, str):
                    doc_val = _decode_unicode_escapes(doc_val)
                item: Dict[str, Any] = {
                    "id": id_,
                    "document": doc_val,
                    "metadata": meta,
                }
                if raw:
                    # 读取原始文档文本（按 metadata.path 与扩展名选择读取方式）
                    path = meta.get("path")
                    raw_text = None
                    if isinstance(path, str) and os.path.exists(path):
                        ext = os.path.splitext(path)[1].lower()
                        try:
                            if ext in (".txt", ".md"):
                                raw_text = _read_text_file(path)
                            elif ext == ".docx":
                                raw_text = _read_docx_file(path)
                        except Exception:
                            raw_text = None
                    if isinstance(raw_text, str):
                        raw_text = _decode_unicode_escapes(raw_text)
                    item["raw_document"] = raw_text
                items.append(item)
            results[p] = items
            logger.info(f"集合实体: {p}/{collection_name} -> {len(items)} 条 (raw={raw})")
        except Exception as e:
            logger.error(f"读取 {p}/{collection_name} 失败: {e}")
            results[p] = [{"error": str(e)}]
    return results


# ---------- Utilities for ingestion ----------
import uuid
import hashlib

try:
    from chromadb.utils.embedding_functions import (
        OpenAIEmbeddingFunction,
        SentenceTransformerEmbeddingFunction,
    )
except Exception:
    OpenAIEmbeddingFunction = None
    SentenceTransformerEmbeddingFunction = None


def _select_embedding_function(backend: str = None, model_name: str = None):
    """选择嵌入函数: backend in {openai, sbert}. 返回 None 则不使用嵌入函数。"""
    backend = (backend or "").strip().lower()
    if backend == "openai" and OpenAIEmbeddingFunction is not None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        return OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name or "text-embedding-3-small")
    if backend in ("sbert", "sentence-transformers") and SentenceTransformerEmbeddingFunction is not None:
        return SentenceTransformerEmbeddingFunction(model_name=model_name or "all-MiniLM-L6-v2")
    # 无有效 backend 或未安装依赖
    return None


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_docx_file(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        # 尝试 docx2txt
        try:
            import docx2txt
            return docx2txt.process(path) or ""
        except Exception:
            return ""


def _split_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    if n == 0:
        return chunks
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


# ---------- DashScope-based embeddings & path helpers ----------
from typing import Optional, Tuple


def _build_dashscope_embeddings(embedding_model: str, dashscope_api_key: Optional[str] = None) -> DashScopeEmbeddings:
    if dashscope_api_key is None:
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise RuntimeError("缺少 DashScope API Key，请设置环境变量 DASHSCOPE_API_KEY 或传入参数。")
    return DashScopeEmbeddings(model=embedding_model, dashscope_api_key=dashscope_api_key)


def _derive_collection_and_persist(vectorstore_dir: str, embedding_model: str, collection_name_base: str = "my_collection") -> Tuple[str, DashScopeEmbeddings, int, str]:
    embeddings = _build_dashscope_embeddings(embedding_model)
    probe_vec = embeddings.embed_query("dimension probe")
    emb_dim = len(probe_vec)
    collection_name = f"{collection_name_base}_dim{emb_dim}"
    model_dir_tag = embedding_model.replace(":", "_").replace("/", "_")
    persist_dir = os.path.join(vectorstore_dir, model_dir_tag)
    return collection_name, embeddings, emb_dim, persist_dir


# ---------- Maintenance operations ----------

def delete_entities_by_source(vectorstore_dir: str, collection_name: str, source_name: str) -> Dict[str, Any]:
    """按 metadata.source 删除实体，返回每个持久化路径的删除数量。"""
    results: Dict[str, Any] = {}
    for p in _persist_paths(vectorstore_dir):
        try:
            client = _build_client(p)
            coll = None
            try:
                coll = client.get_collection(name=collection_name)
            except Exception:
                results[p] = {"deleted": 0, "message": "collection not found"}
                continue
            # 先查出需要删除的 ids
            got = coll.get(where={"source": source_name}, include=["ids"]) or {}
            ids = got.get("ids") or []
            if not ids:
                results[p] = {"deleted": 0, "message": "no match"}
                continue
            coll.delete(ids=ids)
            results[p] = {"deleted": len(ids)}
            logger.info(f"删除实体: {p}/{collection_name} by source={source_name} -> {len(ids)} 条")
        except Exception as e:
            logger.error(f"删除失败 {p}/{collection_name}: {e}")
            results[p] = {"error": str(e)}
    return results


def ingest_knowledge_base(vectorstore_dir: str, collection_name: str, kb_dir: str = "./RAG/knowledge_base", backend: str = None, model_name: str = None, chunk_size: int = 1200, chunk_overlap: int = 200) -> Dict[str, Any]:
    """
    将 knowledge_base 目录中的文档导入到指定 collection。对齐仓库其它脚本：
    - 使用 DashScopeEmbeddings 与 Chroma
    - 集合名追加 _dim{emb_dim} 后缀
    - persist_directory 使用模型专属子目录
    兼容旧参数：使用 model_name 作为 embedding_model。
    """
    results: Dict[str, Any] = {}
    if not os.path.isdir(kb_dir):
        return {"error": f"知识库目录不存在: {kb_dir}"}

    embedding_model = model_name or os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    # 收集文件列表
    files: List[str] = []
    for root, _, fnames in os.walk(kb_dir):
        for fn in fnames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".txt", ".md", ".docx"):
                files.append(os.path.join(root, fn))

    for p in _persist_paths(vectorstore_dir):
        summary = {"files": 0, "chunks": 0, "added": 0, "errors": []}
        try:
            # 对齐其它脚本：按模型维度与模型目录定位目标集合与目录
            collection_name_base = collection_name
            derived_name, embeddings, emb_dim, persist_dir = _derive_collection_and_persist(
                vectorstore_dir=p,
                embedding_model=embedding_model,
                collection_name_base=collection_name_base,
            )

            # 使用 LangChain 的 Chroma（与其它脚本一致）
            vs = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=derived_name,
            )

            for fp in files:
                ext = os.path.splitext(fp)[1].lower()
                text = ""
                if ext in (".txt", ".md"):
                    text = _read_text_file(fp)
                elif ext == ".docx":
                    text = _read_docx_file(fp)
                else:
                    continue
                source_name = os.path.basename(fp)
                chunks = _split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    continue
                ids = []
                metadatas = []
                for idx, ch in enumerate(chunks):
                    base = f"{fp}:{idx}"
                    prefix = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
                    _id = f"{prefix}-{uuid.uuid4()}"
                    ids.append(_id)
                    metadatas.append({
                        "source": source_name,
                        "path": fp,
                        "chunk_index": idx,
                        "type": ext.lstrip("."),
                    })
                try:
                    # 与其它脚本一致：add_texts
                    vs.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
                    summary["files"] += 1
                    summary["chunks"] += len(chunks)
                    summary["added"] += len(chunks)
                    logger.info(f"导入: {p}/{derived_name} <- {source_name} ({len(chunks)} chunks)")
                except Exception as e:
                    logger.error(f"导入失败: {fp}: {e}")
                    summary["errors"].append({"file": fp, "error": str(e)})
        except Exception as e:
            logger.error(f"导入失败路径 {p}: {e}")
            summary["errors"].append({"path": p, "error": str(e)})
        results[p] = summary
    return results


# ---------- Flask endpoints ----------
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# CORS headers for all responses
@app.after_request
def add_cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

# List collections
@app.get('/collections')
def http_collections():
    vectorstore_dir = request.args.get('vectorstore_dir', os.getenv('VECTORSTORE_DIR', './RAG/chroma_db'))
    try:
        data = list_chroma_collections(vectorstore_dir)
        return jsonify({"status": "ok", "vectorstore_dir": vectorstore_dir, "data": data}), 200
    except Exception as e:
        logger.error(f"读取集合失败: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# List entities by collection
@app.get('/entities')
def http_entities():
    vectorstore_dir = request.args.get('vectorstore_dir', os.getenv('VECTORSTORE_DIR', './RAG/chroma_db'))
    collection_name = (os.getenv('COLLECTION_NAME') or '').strip()
    raw_flag = request.args.get('raw', 'false').strip().lower() in ('1', 'true', 'yes')
    if not collection_name:
        return jsonify({"status": "error", "message": "Missing env COLLECTION_NAME"}), 400
    try:
        data = get_collection_entities(vectorstore_dir, collection_name, raw=raw_flag)
        return jsonify({
            "status": "ok",
            "vectorstore_dir": vectorstore_dir,
            "collection_name": collection_name,
            "raw": raw_flag,
            "data": data,
        }), 200
    except Exception as e:
        logger.error(f"读取实体失败: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Delete by source
@app.get('/delete_by_source')
def http_delete_by_source():
    vectorstore_dir = request.args.get('vectorstore_dir', os.getenv('VECTORSTORE_DIR', './RAG/chroma_db'))
    collection_name = (os.getenv('COLLECTION_NAME') or '').strip()
    source_name = request.args.get('source_name')
    if not collection_name:
        return jsonify({"status": "error", "message": "Missing env COLLECTION_NAME"}), 400
    if not source_name:
        return jsonify({"status": "error", "message": "Missing 'source_name'"}), 400
    try:
        data = delete_entities_by_source(vectorstore_dir, collection_name, source_name)
        return jsonify({"status": "ok", "vectorstore_dir": vectorstore_dir, "collection_name": collection_name, "source_name": source_name, "data": data}), 200
    except Exception as e:
        logger.error(f"删除实体失败: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Ingest knowledge base into collection
@app.get('/ingest')
def http_ingest():
    vectorstore_dir = request.args.get('vectorstore_dir', os.getenv('VECTORSTORE_DIR', './RAG/chroma_db'))
    collection_name = (os.getenv('COLLECTION_NAME') or '').strip()
    kb_dir = request.args.get('kb_dir', './RAG/knowledge_base')
    embedding_model = request.args.get('embedding_model', os.getenv('EMBEDDING_MODEL', 'text-embedding-v4'))
    if not collection_name:
        return jsonify({"status": "error", "message": "Missing env COLLECTION_NAME"}), 400
    try:
        data = ingest_knowledge_base(
            vectorstore_dir=vectorstore_dir,
            collection_name=collection_name,
            kb_dir=kb_dir,
            model_name=embedding_model,
        )
        return jsonify({
            "status": "ok",
            "vectorstore_dir": vectorstore_dir,
            "collection_name": collection_name,
            "kb_dir": kb_dir,
            "embedding_model": embedding_model,
            "data": data,
        }), 200
    except Exception as e:
        logger.error(f"导入失败: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.get('/sources')
def http_sources():
    vectorstore_dir = request.args.get('vectorstore_dir', os.getenv('VECTORSTORE_DIR', './RAG/chroma_db'))
    collection_name = (os.getenv('COLLECTION_NAME') or '').strip()
    if not collection_name:
        return jsonify({"status": "error", "message": "Missing env COLLECTION_NAME"}), 400
    try:
        data = list_collection_sources(vectorstore_dir, collection_name)
        return jsonify({
            "status": "ok",
            "vectorstore_dir": vectorstore_dir,
            "collection_name": collection_name,
            "data": data,
        }), 200
    except Exception as e:
        logger.error(f"读取来源失败: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500



# Flask run helper

def run_flask_server(host: str = "0.0.0.0", port: int = int(os.getenv("MAINTAIN_PORT", "8001"))):
    logger.info(f"Flask Maintain 服务已启动: http://localhost:{port}/collections?vectorstore_dir=./RAG/chroma_db")
    logger.info(f"实体查询示例: http://localhost:{port}/entities?collection_name=my_collection_dim1024&vectorstore_dir=./RAG/chroma_db")
    logger.info(f"删除示例: http://localhost:{port}/delete_by_source?collection_name=my_collection_dim1024&source_name=foo.txt&vectorstore_dir=./RAG/chroma_db")
    logger.info(f"导入示例: http://localhost:{port}/ingest?collection_name=my_collection_dim1024&kb_dir=./RAG/knowledge_base&backend=sbert&model=all-MiniLM-L6-v2")
    app.run(host=host, port=port)


if __name__ == "__main__":
    run_flask_server()
