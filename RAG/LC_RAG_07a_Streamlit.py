import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# 加载环境变量
load_dotenv()

# 配置页面
st.set_page_config(
    page_title="DeepSeek 问答应用",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置页面标题和说明
st.title("🤖 DeepSeek 智能问答应用")
st.markdown("使用 LangChain 1.0 和 Streamlit 构建的对话系统")

# 侧边栏配置
with st.sidebar:
    st.header("配置选项")
    
    # API 密钥配置
    deepseek_api_key = st.text_input(
        "DeepSeek API Key",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        type="password",
        help="请输入您的 DeepSeek API Key"
    )
    
    # 清除对话历史按钮
    if st.button("清除对话历史", type="secondary"):
        st.session_state["messages"] = []
        st.rerun()

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 显示对话历史
for message in st.session_state["messages"]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 处理用户输入
user_input = st.chat_input("请输入您的问题...")

if user_input:
    # 验证API密钥
    if not deepseek_api_key:
        st.error("请在侧边栏配置您的 DeepSeek API Key")
        st.stop()
    
    # 添加用户消息到会话历史
    st.session_state["messages"].append(HumanMessage(content=user_input))
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 显示助手思考中状态
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            chat_model=init_chat_model(model="deepseek:deepseek-chat")
            
            # 创建提示模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个智能助手，使用中文回答用户问题，保持友好和专业。"),
                *st.session_state["messages"]
            ])
            
            # 创建链
            chain = prompt | chat_model | StrOutputParser()
            
            try:
                # 生成回答
                response = chain.invoke({})
                
                # 显示回答
                st.markdown(response)
                
                # 添加助手消息到会话历史
                st.session_state["messages"].append(AIMessage(content=response))
                
            except Exception as e:
                st.error(f"请求失败: {str(e)}")
                st.error("请检查您的API密钥或网络连接")

# 底部信息
st.markdown("---")
st.markdown("💡 提示：在侧边栏可以配置API密钥。")