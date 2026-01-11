import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


def get_llm_response(user_input: str, expert_type: str) -> str:
    """
    入力テキストと専門家の種類を受け取り、LLMからの回答を返す関数
    
    Args:
        user_input: ユーザーからの入力テキスト
        expert_type: 選択された専門家の種類
    
    Returns:
        LLMからの回答テキスト
    """
    # 専門家の種類に応じてシステムメッセージを設定
    system_messages = {
        "医療専門家": "あなたは経験豊富な医療専門家です。医学的な質問に対して、正確で分かりやすい回答を提供してください。",
        "プログラミング専門家": "あなたは熟練したプログラミングエキスパートです。技術的な質問に対して、実践的で詳細なアドバイスを提供してください。",
        "ビジネスコンサルタント": "あなたは経験豊富なビジネスコンサルタントです。ビジネス戦略や経営に関する質問に対して、実用的な助言を提供してください。"
    }
    
    # ChatOpenAIインスタンスを作成
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    
    # メッセージを構築
    messages = [
        SystemMessage(content=system_messages[expert_type]),
        HumanMessage(content=user_input)
    ]
    
    # LLMに送信して回答を取得
    response = chat.invoke(messages)
    
    return response.content


# Streamlitアプリケーションのメイン部分
st.title("🤖 AI専門家チャットアプリ")

st.markdown("""
### 📋 アプリケーション概要
このアプリケーションでは、異なる分野の専門家としてAIと会話することができます。
専門家の種類を選択し、質問を入力すると、その分野の専門家として回答を提供します。

### 💡 使い方
1. **専門家の種類を選択**: ラジオボタンから相談したい専門家の種類を選んでください
2. **質問を入力**: テキストエリアに質問や相談内容を入力してください
3. **回答を取得**: 「回答を取得」ボタンをクリックすると、AIが専門家として回答します
""")

st.divider()

# 専門家の種類を選択するラジオボタン
expert_type = st.radio(
    "相談したい専門家の種類を選択してください:",
    ["医療専門家", "プログラミング専門家", "ビジネスコンサルタント"],
    horizontal=True
)

# ユーザー入力フォーム
user_input = st.text_area(
    "質問を入力してください:",
    height=150,
    placeholder="ここに質問を入力してください..."
)

# 送信ボタン
if st.button("回答を取得", type="primary"):
    if user_input.strip():
        with st.spinner("AI専門家が回答を生成中..."):
            try:
                # LLMから回答を取得
                response = get_llm_response(user_input, expert_type)
                
                # 回答を表示
                st.success("回答が生成されました！")
                st.markdown("### 💬 回答:")
                st.markdown(response)
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                st.info("OpenAI APIキーが設定されているか確認してください。")
    else:
        st.warning("質問を入力してください。")