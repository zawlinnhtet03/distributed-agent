"""ModelFactory - Centralized model configuration for ADK agents.

This factory provides pre-configured LiteLLM model strings and parameters
for use with Google ADK's LlmAgent classes. Eliminates repetitive API
configuration across multiple agents.
"""

from dataclasses import dataclass
import os
from typing import Any

import asyncio
import random
import time

from google.adk.models.lite_llm import LiteLLMClient, LiteLlm

from litellm import acompletion, completion

from app.config import ModelTier, get_settings


class RetryingLiteLLMClient(LiteLLMClient):
    def __init__(
        self,
        num_retries: int,
        base_delay_s: float,
        max_delay_s: float,
    ) -> None:
        super().__init__()
        self._num_retries = max(0, int(num_retries))
        self._base_delay_s = float(base_delay_s)
        self._max_delay_s = float(max_delay_s)

    def _is_retryable(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            s in msg
            for s in [
                "service tier capacity exceeded",
                "service_tier_capacity_exceeded",
                "rate limit",
                "ratelimit",
                "code\":\"3505",
                "code=3505",
            ]
        )

    def _delay_for_attempt(self, attempt: int) -> float:
        base = max(0.0, self._base_delay_s) * (2 ** max(0, attempt))
        jitter = random.uniform(0.0, 0.25 * max(1.0, base))
        return min(self._max_delay_s, base + jitter) if self._max_delay_s > 0 else (base + jitter)

    async def acompletion(self, model, messages, tools, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self._num_retries + 1):
            try:
                return await acompletion(model=model, messages=messages, tools=tools, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self._num_retries or not self._is_retryable(exc):
                    raise
                await asyncio.sleep(self._delay_for_attempt(attempt))
        raise last_exc  # type: ignore[misc]

    def completion(self, model, messages, tools, stream=False, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self._num_retries + 1):
            try:
                return completion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    stream=stream,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self._num_retries or not self._is_retryable(exc):
                    raise
                time.sleep(self._delay_for_attempt(attempt))
        raise last_exc  # type: ignore[misc]


@dataclass
class ModelConfig:
    """Configuration container for a model instance."""

    model_id: str
    temperature: float
    max_tokens: int
    api_key: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LiteLLM kwargs."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class ModelFactory:
    """Factory class for creating LiteLLM model instances (Mistral).

    Usage:
        # Get default model for an agent
        model = ModelFactory.create()

        # Get fast model for quick tasks
        fast_model = ModelFactory.create(tier="fast")

        # Custom configuration
        custom_model = ModelFactory.create(
            tier="default",
            temperature=0.3,
            max_tokens=2048
        )

        # Use with ADK LlmAgent
        agent = LlmAgent(
            name="my_agent",
            model=ModelFactory.create(),
            ...
        )
    """

    _settings = None

    @classmethod
    def _get_settings(cls):
        """Lazy load settings instance."""
        if cls._settings is None:
            cls._settings = get_settings()
        return cls._settings

    @classmethod
    def get_model_id(cls, tier: ModelTier = "default") -> str:
        """
        Get the model identifier string for a given tier.

        Args:
            tier: Model tier - "default", "fast", or "embedding"

        Returns:
            Model identifier string (e.g., "mistral/mistral-medium-latest")
        """
        settings = cls._get_settings()

        # Prefer Mistral when available.
        if settings.mistral_api_key:
            return settings.model_ids.get(tier, settings.default_model)

        # Fallback to Groq if Mistral key is not configured.
        if settings.groq_api_key:
            default_model = os.getenv("GROQ_DEFAULT_MODEL", "groq/llama-3.1-8b-instant")
            fast_model = os.getenv("GROQ_FAST_MODEL", default_model)
            model_map = {
                "default": default_model,
                "fast": fast_model,
                "embedding": fast_model,
            }
            return model_map.get(tier, default_model)

        raise RuntimeError("No model API key configured. Set MISTRAL_API_KEY or GROQ_API_KEY.")

    @classmethod
    def get_config(
        cls,
        tier: ModelTier = "default",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelConfig:
        """
        Get a ModelConfig instance with all parameters.

        Args:
            tier: Model tier selection
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            ModelConfig instance with all parameters
        """
        settings = cls._get_settings()
        api_key = settings.mistral_api_key or settings.groq_api_key
        if not api_key:
            raise RuntimeError("No model API key configured. Set MISTRAL_API_KEY or GROQ_API_KEY.")

        return ModelConfig(
            model_id=cls.get_model_id(tier),
            temperature=temperature or settings.default_temperature,
            max_tokens=max_tokens or settings.default_max_tokens,
            api_key=api_key,
        )

    @classmethod
    def create(
        cls,
        tier: ModelTier = "default",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LiteLlm:
        """Create a LiteLLM model instance for ADK agents.

        This is the primary method to use when creating agents. It returns
        a fully configured LiteLlm instance (using Mistral) ready for use
        with LlmAgent.

        Args:
            tier: Model tier - "default" for complex tasks, "fast" for quick responses
            temperature: Override default temperature (0.0-1.0)
            max_tokens: Override default max output tokens

        Returns:
            Configured LiteLlm instance for use with ADK LlmAgent

        Example:
            agent = LlmAgent(
                name="retrieval_agent",
                model=ModelFactory.create(tier="default", temperature=0.3),
                instruction="You are a retrieval specialist...",
            )
        """
        config = cls.get_config(tier, temperature, max_tokens)
        settings = cls._get_settings()

        return LiteLlm(
            model=config.model_id,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            llm_client=RetryingLiteLLMClient(
                num_retries=settings.llm_num_retries,
                base_delay_s=settings.llm_retry_base_delay_s,
                max_delay_s=settings.llm_retry_max_delay_s,
            ),
        )

    @classmethod
    def create_fast(cls, temperature: float = 0.5) -> LiteLlm:
        """Shorthand for creating a fast-tier model."""
        return cls.create(tier="fast", temperature=temperature)

    @classmethod
    def create_precise(cls, max_tokens: int = 2048) -> LiteLlm:
        """Shorthand for creating a precise, low-temperature model."""
        return cls.create(tier="default", temperature=0.1, max_tokens=max_tokens)
