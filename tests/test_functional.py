import unittest
from unittest.mock import MagicMock, patch
from core.agent import ClinicalAssistant


class TestFunctional(unittest.TestCase):
    @patch("core.agent.ChatOpenAI")
    @patch("core.agent.MedicalSearchTool")
    def test_agent_initialization(self, mock_tool, mock_llm):
        # Mock vector store
        mock_vs = MagicMock()
        assistant = ClinicalAssistant(mock_vs)

        self.assertIsNotNone(assistant.llm)
        self.assertEqual(len(assistant.tools), 1)

    @patch("core.agent.ChatOpenAI")
    @patch("core.agent.create_openai_functions_agent")
    @patch("core.agent.AgentExecutor")
    def test_get_executor(self, mock_executor, mock_agent, mock_llm):
        mock_vs = MagicMock()
        assistant = ClinicalAssistant(mock_vs)

        executor = assistant.get_executor("identity info")
        self.assertIsNotNone(executor)
        # Verify it was called with tools
        mock_executor.assert_called()


if __name__ == "__main__":
    unittest.main()
