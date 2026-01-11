"""
app.pyのテストコード
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app import get_llm_response


class TestGetLLMResponse:
    """get_llm_response関数のテストクラス"""
    
    @pytest.fixture
    def mock_chat_response(self):
        """ChatOpenAIのモックレスポンスを返すフィクスチャ"""
        mock_response = Mock()
        mock_response.content = "これはテスト用の回答です。"
        return mock_response
    
    @patch('app.ChatOpenAI')
    def test_medical_expert_response(self, mock_chat_openai, mock_chat_response):
        """医療専門家としての回答をテスト"""
        # モックの設定
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_chat_response
        mock_chat_openai.return_value = mock_instance
        
        # テスト実行
        result = get_llm_response("頭痛がします", "医療専門家")
        
        # 検証
        assert result == "これはテスト用の回答です。"
        mock_chat_openai.assert_called_once_with(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # invokeに渡されたメッセージを確認
        call_args = mock_instance.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert "医療専門家" in call_args[0].content
        assert isinstance(call_args[1], HumanMessage)
        assert call_args[1].content == "頭痛がします"
    
    @patch('app.ChatOpenAI')
    def test_programming_expert_response(self, mock_chat_openai, mock_chat_response):
        """プログラミング専門家としての回答をテスト"""
        # モックの設定
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_chat_response
        mock_chat_openai.return_value = mock_instance
        
        # テスト実行
        result = get_llm_response("Pythonの関数について教えてください", "プログラミング専門家")
        
        # 検証
        assert result == "これはテスト用の回答です。"
        
        # invokeに渡されたメッセージを確認
        call_args = mock_instance.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert "プログラミングエキスパート" in call_args[0].content
        assert isinstance(call_args[1], HumanMessage)
        assert call_args[1].content == "Pythonの関数について教えてください"
    
    @patch('app.ChatOpenAI')
    def test_business_consultant_response(self, mock_chat_openai, mock_chat_response):
        """ビジネスコンサルタントとしての回答をテスト"""
        # モックの設定
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_chat_response
        mock_chat_openai.return_value = mock_instance
        
        # テスト実行
        result = get_llm_response("新規事業の立ち上げ方を教えてください", "ビジネスコンサルタント")
        
        # 検証
        assert result == "これはテスト用の回答です。"
        
        # invokeに渡されたメッセージを確認
        call_args = mock_instance.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], SystemMessage)
        assert "ビジネスコンサルタント" in call_args[0].content
        assert isinstance(call_args[1], HumanMessage)
        assert call_args[1].content == "新規事業の立ち上げ方を教えてください"
    
    @patch('app.ChatOpenAI')
    def test_function_calls_chatgpt_correctly(self, mock_chat_openai):
        """ChatOpenAIが正しいパラメータで呼び出されることをテスト"""
        # モックの設定
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.content = "回答"
        mock_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_instance
        
        # テスト実行
        get_llm_response("テスト質問", "医療専門家")
        
        # ChatOpenAIが正しいパラメータで初期化されたことを確認
        mock_chat_openai.assert_called_once_with(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # invokeが呼び出されたことを確認
        assert mock_instance.invoke.called
    
    @patch('app.ChatOpenAI')
    def test_empty_input(self, mock_chat_openai, mock_chat_response):
        """空の入力に対する動作をテスト"""
        # モックの設定
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_chat_response
        mock_chat_openai.return_value = mock_instance
        
        # テスト実行
        result = get_llm_response("", "医療専門家")
        
        # 検証（空の入力でも処理されることを確認）
        assert result == "これはテスト用の回答です。"
        
        # 空の入力がHumanMessageとして渡されたことを確認
        call_args = mock_instance.invoke.call_args[0][0]
        assert call_args[1].content == ""
    
    @patch('app.ChatOpenAI')
    def test_long_input(self, mock_chat_openai, mock_chat_response):
        """長い入力に対する動作をテスト"""
        # モックの設定
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_chat_response
        mock_chat_openai.return_value = mock_instance
        
        # 長い入力テキスト
        long_input = "これは非常に長い質問です。" * 100
        
        # テスト実行
        result = get_llm_response(long_input, "プログラミング専門家")
        
        # 検証
        assert result == "これはテスト用の回答です。"
        
        # 長い入力が正しく渡されたことを確認
        call_args = mock_instance.invoke.call_args[0][0]
        assert call_args[1].content == long_input
    
    @patch('app.ChatOpenAI')
    def test_all_expert_types_have_system_messages(self, mock_chat_openai, mock_chat_response):
        """すべての専門家タイプに対してシステムメッセージが定義されていることをテスト"""
        # モックの設定
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_chat_response
        mock_chat_openai.return_value = mock_instance
        
        expert_types = ["医療専門家", "プログラミング専門家", "ビジネスコンサルタント"]
        
        for expert_type in expert_types:
            # テスト実行
            result = get_llm_response("テスト質問", expert_type)
            
            # 検証
            assert result == "これはテスト用の回答です。"
            
            # システムメッセージが設定されていることを確認
            call_args = mock_instance.invoke.call_args[0][0]
            assert isinstance(call_args[0], SystemMessage)
            assert len(call_args[0].content) > 0


class TestIntegration:
    """統合テスト（実際のAPIを呼び出さない範囲での統合）"""
    
    @patch('app.ChatOpenAI')
    def test_workflow_simulation(self, mock_chat_openai):
        """ユーザーワークフロー全体をシミュレーション"""
        # モックの設定
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.content = "専門的な回答がここに入ります。"
        mock_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_instance
        
        # シナリオ1: 医療専門家への質問
        user_input_1 = "風邪の症状について教えてください"
        expert_type_1 = "医療専門家"
        response_1 = get_llm_response(user_input_1, expert_type_1)
        assert response_1 == "専門的な回答がここに入ります。"
        
        # シナリオ2: プログラミング専門家への質問
        user_input_2 = "Pythonのリスト内包表記について"
        expert_type_2 = "プログラミング専門家"
        response_2 = get_llm_response(user_input_2, expert_type_2)
        assert response_2 == "専門的な回答がここに入ります。"
        
        # シナリオ3: ビジネスコンサルタントへの質問
        user_input_3 = "マーケティング戦略について"
        expert_type_3 = "ビジネスコンサルタント"
        response_3 = get_llm_response(user_input_3, expert_type_3)
        assert response_3 == "専門的な回答がここに入ります。"
        
        # すべてのシナリオでinvokeが呼び出されたことを確認
        assert mock_instance.invoke.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
