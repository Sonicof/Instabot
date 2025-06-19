import pytest
from rag_system import RAGSystem

class TestRAGSystem:
    @pytest.fixture(scope="class")
    def rag_system(self):
        """Setup RAG system for testing"""
        rag = RAGSystem()
        
        # Add test data
        test_text = """
        Monopoly is a board game where players buy and trade properties.
        The goal is to bankrupt other players by collecting rent.
        Players move around the board by rolling dice.
        There are 40 spaces on the Monopoly board.
        """
        
        rag.add_text(test_text, "monopoly_rules.txt")
        return rag
    
    def test_basic_query(self, rag_system):
        """Test basic functionality"""
        result = rag_system.query("What is Monopoly?")
        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert "monopoly" in result["answer"].lower()
    
    def test_specific_fact(self, rag_system):
        """Test specific fact retrieval"""
        result = rag_system.query("How many spaces are on the board?")
        assert "40" in result["answer"] or "forty" in result["answer"].lower()
    
    def test_llm_evaluation(self, rag_system):
        """Test using LLM for answer evaluation"""
        question = "What is the goal of Monopoly?"
        result = rag_system.query(question)
        expected = "To bankrupt other players"
        
        # Use LLM to evaluate equivalence
        is_correct = rag_system.llm_client.evaluate_answer(
            question, expected, result["answer"]
        )
        assert is_correct, f"Expected answer about bankrupting players, got: {result['answer']}"
    
    def test_negative_case(self, rag_system):
        """Test that system doesn't make up information"""
        result = rag_system.query("What is the capital of Mars?")
        answer_lower = result["answer"].lower()
        
        # Should indicate lack of information
        assert any(phrase in answer_lower for phrase in [
            "don't have enough information",
            "not in the context",
            "cannot answer",
            "no information"
        ])
