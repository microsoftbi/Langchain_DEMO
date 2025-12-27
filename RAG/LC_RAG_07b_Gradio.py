import os
import gradio as gr
from dotenv import load_dotenv
from LC_RAG_02_RecallTest import recall

# 加载环境变量
load_dotenv()

def run_recall_test(query, top_k):
    """运行召回测试并返回结果"""
    try:
        # 调用recall函数，但需要捕获其输出
        import io
        from contextlib import redirect_stdout
        
        # 创建一个字符串IO对象来捕获输出
        f = io.StringIO()
        with redirect_stdout(f):
            recall(
                query=query,
                top_k=top_k,
                vectorstore_dir="./RAG/chroma_db",
                embedding_model="text-embedding-v4"
            )
        
        # 获取捕获的输出
        output = f.getvalue()
        return output
    except Exception as e:
        return f"错误: {str(e)}"

# 创建界面
with gr.Blocks(title="RAG召回测试系统", theme=gr.themes.Soft()) as demo:
    # 页面标题
    gr.Markdown("# RAG召回测试系统")
    gr.Markdown("使用Gradio构建的RAG召回测试界面，调用LC_RAG_02_RecallTest.py实现相似度检索")
    
    with gr.Row():
        # 左侧：输入区域
        with gr.Column(scale=1):
            gr.Markdown("## 查询设置")
            
            # 查询文本输入
            query_input = gr.Textbox(
                label="查询文本",
                placeholder="请输入您要检索的问题或关键词",
                lines=3,
                info="例如: '什么是未成年?' 或 '哪些节假日应该安排休假?'"
            )
            
            # Top-K设置
            top_k_slider = gr.Slider(
                label="Top-K检索数量",
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                info="设置返回的最相似文档数量"
            )
            
            # 提交按钮
            submit_btn = gr.Button(
                "执行召回测试",
                variant="primary",
                size="lg"
            )
            
            # 重置按钮
            clear_btn = gr.Button("重置", variant="secondary")
        
        # 右侧：输出区域
        with gr.Column(scale=2):
            gr.Markdown("## 召回结果")
            
            # 结果输出
            result_output = gr.Textbox(
                label="相似度检索结果",
                lines=20,
                interactive=False,
                placeholder="召回结果将显示在这里..."
            )
            
            # 状态信息
            status_info = gr.Markdown("状态: 就绪")
    
    # 按钮事件
    def on_submit(query, top_k):
        if not query:
            return "请输入查询文本", "状态: 错误 - 查询文本不能为空"
        
        status_info.value = "状态: 正在执行召回测试..."
        result = run_recall_test(query, top_k)
        return result, "状态: 测试完成"
    
    def on_clear():
        return "", 5, "", "状态: 已重置"
    
    # 绑定事件
    submit_btn.click(
        fn=lambda query, top_k: on_submit(query, top_k),
        inputs=[query_input, top_k_slider],
        outputs=[result_output, status_info]
    )
    
    clear_btn.click(
        fn=on_clear,
        inputs=[],
        outputs=[query_input, top_k_slider, result_output, status_info]
    )
    
    # 示例查询
    gr.Markdown("## 示例查询")
    with gr.Row():
        example1 = gr.Button("什么是未成年?")
        example2 = gr.Button("哪些节假日应该安排休假?")
        example3 = gr.Button("足球比赛的基本规则是什么?")
    
    def set_example(example_text):
        return example_text, 5, "", "状态: 就绪"
    
    example1.click(fn=lambda: set_example("什么是未成年?"), inputs=[], outputs=[query_input, top_k_slider, result_output, status_info])
    example2.click(fn=lambda: set_example("哪些节假日应该安排休假?"), inputs=[], outputs=[query_input, top_k_slider, result_output, status_info])
    example3.click(fn=lambda: set_example("足球比赛的基本规则是什么?"), inputs=[], outputs=[query_input, top_k_slider, result_output, status_info])

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )