"""Unit tests for AdaptiveJudgeModule."""

import pytest
import dspy
from unittest.mock import Mock, patch, MagicMock
from src.factchecker.modules.adaptive_judge_module import AdaptiveJudgeModule


class TestAdaptiveJudgeModule:
    """Test suite for AdaptiveJudgeModule."""

    def test_initialization_default_params(self):
        """Test module initializes with default parameters."""
        module = AdaptiveJudgeModule()

        assert module.confidence_threshold == 0.7
        assert module.enable_fallback is True
        assert module.max_judge_iterations == 3
        assert module.max_page_visits == 3
        assert module._pipeline is None  # Lazy initialization

    def test_initialization_custom_params(self):
        """Test module initializes with custom parameters."""
        module = AdaptiveJudgeModule(
            confidence_threshold=0.5,
            enable_fallback=False,
            max_judge_iterations=5,
            max_page_visits=2
        )

        assert module.confidence_threshold == 0.5
        assert module.enable_fallback is False
        assert module.max_judge_iterations == 5
        assert module.max_page_visits == 2

    def test_initialization_invalid_confidence_threshold(self):
        """Test that invalid confidence threshold raises ValueError."""
        with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
            AdaptiveJudgeModule(confidence_threshold=1.5)

        with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
            AdaptiveJudgeModule(confidence_threshold=-0.1)

    @patch('src.factchecker.modules.adaptive_judge_module.JudgeModule')
    def test_no_fallback_high_confidence_unsupported(self, mock_judge_class):
        """Test no fallback when confidence is above threshold."""
        # Setup mock
        mock_judge = Mock()
        mock_judge_class.return_value = mock_judge

        mock_result = Mock()
        mock_result.overall_verdict = "CONTAINS_UNSUPPORTED_CLAIMS"
        mock_result.confidence = 0.8  # Above threshold
        mock_result.reasoning = "High confidence reasoning"
        mock_judge.return_value = mock_result

        # Test
        module = AdaptiveJudgeModule(confidence_threshold=0.7)
        result = module(statement="Test statement")

        # Verify
        assert result.overall_verdict == "CONTAINS_UNSUPPORTED_CLAIMS"
        assert result.confidence == 0.8
        assert result.fallback_triggered is False
        assert result.reasoning == "High confidence reasoning"
        assert not hasattr(result, 'claims')  # No pipeline data

    @patch('src.factchecker.modules.adaptive_judge_module.JudgeModule')
    def test_no_fallback_supported_verdict(self, mock_judge_class):
        """Test no fallback for SUPPORTED verdict regardless of confidence."""
        mock_judge = Mock()
        mock_judge_class.return_value = mock_judge

        mock_result = Mock()
        mock_result.overall_verdict = "SUPPORTED"
        mock_result.confidence = 0.3  # Low confidence, but SUPPORTED
        mock_result.reasoning = "Supported reasoning"
        mock_judge.return_value = mock_result

        module = AdaptiveJudgeModule(confidence_threshold=0.7)
        result = module(statement="Test statement")

        assert result.overall_verdict == "SUPPORTED"
        assert result.confidence == 0.3
        assert result.fallback_triggered is False

    @patch('src.factchecker.modules.adaptive_judge_module.JudgeModule')
    def test_no_fallback_refuted_verdict(self, mock_judge_class):
        """Test no fallback for CONTAINS_REFUTED_CLAIMS verdict."""
        mock_judge = Mock()
        mock_judge_class.return_value = mock_judge

        mock_result = Mock()
        mock_result.overall_verdict = "CONTAINS_REFUTED_CLAIMS"
        mock_result.confidence = 0.3  # Low confidence, but refuted
        mock_result.reasoning = "Refuted reasoning"
        mock_judge.return_value = mock_result

        module = AdaptiveJudgeModule(confidence_threshold=0.7)
        result = module(statement="Test statement")

        assert result.overall_verdict == "CONTAINS_REFUTED_CLAIMS"
        assert result.confidence == 0.3
        assert result.fallback_triggered is False

    @patch('src.factchecker.modules.adaptive_judge_module.FactCheckerPipeline')
    @patch('src.factchecker.modules.adaptive_judge_module.JudgeModule')
    def test_fallback_triggered(self, mock_judge_class, mock_pipeline_class):
        """Test fallback is triggered when conditions are met."""
        # Setup judge mock
        mock_judge = Mock()
        mock_judge_class.return_value = mock_judge

        mock_judge_result = Mock()
        mock_judge_result.overall_verdict = "CONTAINS_UNSUPPORTED_CLAIMS"
        mock_judge_result.confidence = 0.5  # Below threshold
        mock_judge_result.reasoning = "Uncertain"
        mock_judge.return_value = mock_judge_result

        # Setup pipeline mock
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        mock_pipeline_result = Mock()
        mock_pipeline_result.overall_verdict = "SUPPORTED"
        mock_pipeline_result.confidence = 0.95
        mock_pipeline_result.reasoning = "Verified with research"
        mock_pipeline_result.claims = ["claim1", "claim2"]
        mock_pipeline_result.claim_results = [Mock(), Mock()]
        mock_pipeline.return_value = mock_pipeline_result

        # Test
        module = AdaptiveJudgeModule(confidence_threshold=0.7)
        result = module(statement="Test statement")

        # Verify fallback triggered
        assert result.fallback_triggered is True
        assert result.overall_verdict == "SUPPORTED"
        assert result.confidence == 0.95
        assert result.reasoning == "Verified with research"
        assert hasattr(result, 'claims')
        assert result.claims == ["claim1", "claim2"]

        # Verify pipeline was called
        mock_pipeline.assert_called_once_with(statement="Test statement")

    @patch('src.factchecker.modules.adaptive_judge_module.FactCheckerPipeline')
    @patch('src.factchecker.modules.adaptive_judge_module.JudgeModule')
    def test_fallback_disabled(self, mock_judge_class, mock_pipeline_class):
        """Test no fallback when enable_fallback=False."""
        mock_judge = Mock()
        mock_judge_class.return_value = mock_judge

        mock_judge_result = Mock()
        mock_judge_result.overall_verdict = "CONTAINS_UNSUPPORTED_CLAIMS"
        mock_judge_result.confidence = 0.3  # Well below threshold
        mock_judge_result.reasoning = "Very uncertain"
        mock_judge.return_value = mock_judge_result

        # Test with fallback disabled
        module = AdaptiveJudgeModule(
            confidence_threshold=0.7,
            enable_fallback=False  # Disabled
        )
        result = module(statement="Test statement")

        # Verify no fallback despite low confidence
        assert result.fallback_triggered is False
        assert result.overall_verdict == "CONTAINS_UNSUPPORTED_CLAIMS"
        assert result.confidence == 0.3

        # Verify pipeline was never initialized
        mock_pipeline_class.assert_not_called()

    @patch('src.factchecker.modules.adaptive_judge_module.FactCheckerPipeline')
    @patch('src.factchecker.modules.adaptive_judge_module.JudgeModule')
    def test_lazy_pipeline_initialization(self, mock_judge_class, mock_pipeline_class):
        """Test that pipeline is only initialized when needed."""
        mock_judge = Mock()
        mock_judge_class.return_value = mock_judge

        # First call: no fallback needed
        mock_judge_result = Mock()
        mock_judge_result.overall_verdict = "SUPPORTED"
        mock_judge_result.confidence = 0.9
        mock_judge_result.reasoning = "Clear"
        mock_judge.return_value = mock_judge_result

        module = AdaptiveJudgeModule()
        result = module(statement="Test 1")

        # Pipeline not initialized yet
        assert module._pipeline is None
        mock_pipeline_class.assert_not_called()

        # Second call: fallback needed
        mock_judge_result.overall_verdict = "CONTAINS_UNSUPPORTED_CLAIMS"
        mock_judge_result.confidence = 0.3

        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline_result = Mock()
        mock_pipeline_result.overall_verdict = "SUPPORTED"
        mock_pipeline_result.confidence = 0.9
        mock_pipeline_result.reasoning = "Researched"
        mock_pipeline_result.claims = []
        mock_pipeline_result.claim_results = []
        mock_pipeline.return_value = mock_pipeline_result

        result = module(statement="Test 2")

        # Now pipeline is initialized
        assert module._pipeline is not None
        mock_pipeline_class.assert_called_once()

    def test_confidence_threshold_boundary(self):
        """Test behavior at confidence threshold boundary."""
        with patch('src.factchecker.modules.adaptive_judge_module.JudgeModule') as mock_judge_class:
            mock_judge = Mock()
            mock_judge_class.return_value = mock_judge

            module = AdaptiveJudgeModule(confidence_threshold=0.7)

            # Test exactly at threshold (should NOT trigger fallback)
            mock_result = Mock()
            mock_result.overall_verdict = "CONTAINS_UNSUPPORTED_CLAIMS"
            mock_result.confidence = 0.7  # Exactly at threshold
            mock_result.reasoning = "At threshold"
            mock_judge.return_value = mock_result

            result = module(statement="Test")
            assert result.fallback_triggered is False  # Equal to threshold = no fallback

            # Test just below threshold (should trigger fallback)
            with patch('src.factchecker.modules.adaptive_judge_module.FactCheckerPipeline') as mock_pipeline_class:
                mock_result.confidence = 0.699  # Just below threshold

                mock_pipeline = Mock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline_result = Mock()
                mock_pipeline_result.overall_verdict = "SUPPORTED"
                mock_pipeline_result.confidence = 0.9
                mock_pipeline_result.reasoning = "Researched"
                mock_pipeline_result.claims = []
                mock_pipeline_result.claim_results = []
                mock_pipeline.return_value = mock_pipeline_result

                result = module(statement="Test")
                assert result.fallback_triggered is True


class TestAdaptiveJudgeModuleIntegration:
    """Integration tests that test with real DSPy components (mocked LLM)."""

    @pytest.mark.skipif(
        not hasattr(dspy, 'LM'),
        reason="Requires DSPy with LM support"
    )
    def test_end_to_end_no_fallback(self):
        """Test end-to-end with mocked LLM responses (no fallback)."""
        # This would require setting up a mock LLM
        # Skipping for now as it requires more complex DSPy mocking
        pass

    @pytest.mark.skipif(
        not hasattr(dspy, 'LM'),
        reason="Requires DSPy with LM support"
    )
    def test_end_to_end_with_fallback(self):
        """Test end-to-end with mocked LLM responses (with fallback)."""
        # This would require setting up a mock LLM
        # Skipping for now as it requires more complex DSPy mocking
        pass
