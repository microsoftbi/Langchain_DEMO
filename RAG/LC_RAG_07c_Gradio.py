import os
import gradio as gr
from dotenv import load_dotenv
from LC_RAG_03_QA import answer_question

# 加载环境变量
load_dotenv()

def run_qa(question, top_k=5):
    """运行QA并返回结果"""
    try:
        answer, sources = answer_question(
            question=question,
            top_k=top_k,
            vectorstore_dir="./RAG/chroma_db",
            embedding_model="text-embedding-v4"
        )
        
        # 格式化结果
        result = f"# 🎯 答案\n\n{answer}\n"
        
        if sources:
            result += "\n# 📚 参考来源\n"
            for source in sources:
                result += f"- {source}\n"
        
        return result
    except Exception as e:
        return f"❌ 错误: {str(e)}"

# 创建Gradio Interface
iface = gr.Interface(
    fn=run_qa,
    inputs=[
        gr.Textbox(
            label="问题",
            placeholder="请输入您的问题...",
            lines=3,
            info="例如: '哪些节假日应该安排休假？' 或 '什么是未成年？'"
        ),
        gr.Slider(
            label="Top-K检索数量",
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            info="设置返回的最相似文档数量"
        )
    ],
    outputs=gr.Markdown(
        label="回答结果"
    ),
    title="🤖 RAG问答系统",
    description="基于LangChain和Gradio构建的RAG问答系统，使用向量数据库进行知识检索",
    examples=[
        ["哪些节假日应该安排休假？", 5],
        ["什么是未成年？", 3],
        ["足球比赛的基本规则是什么？", 4]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# 启动应用
if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=False
    )