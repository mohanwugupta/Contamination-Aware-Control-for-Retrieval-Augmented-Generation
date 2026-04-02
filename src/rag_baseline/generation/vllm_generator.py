"""Generator layer (PRD 1 §9.5).

Wraps a vLLM-served model for answer generation.

Two modes:
  1. VLLMGenerator — talks to a *running* vLLM OpenAI-compatible server
     (default for cluster: SLURM script starts server, pipeline is the client).
  2. InProcessVLLMGenerator — loads the model in-process via ``vllm.LLM``
     (mirrors God's Reach Qwen72BProvider; useful for single-shot jobs).
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Raw result from the generator."""

    text: str
    logprobs: list[float] | None = None
    finish_reason: str = ""
    usage: dict | None = None


class BaseGenerator(ABC):
    """Abstract base class for generators."""

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """Generate a response for a prompt.

        Args:
            prompt: The full prompt text.

        Returns:
            GenerationResult with raw text and optional metadata.
        """
        ...


# ---------------------------------------------------------------------------
# Mode 1: External vLLM server (OpenAI-compatible API)
# ---------------------------------------------------------------------------


class VLLMGenerator(BaseGenerator):
    """Generator using vLLM's OpenAI-compatible API.

    Expects a running vLLM server (started by SLURM script or manually).
    Includes health-check, connection retry, and timeout logic so that
    cluster jobs surface clear errors instead of cryptic connection-refused.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-32B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: str = "EMPTY",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._client = None

    # -- health check -------------------------------------------------------

    def health_check(self) -> bool:
        """Verify the vLLM server is reachable and serving the expected model.

        Returns ``True`` if the server responds on ``/v1/models``.
        """
        import urllib.request
        import urllib.error

        # Strip "/v1" suffix to get the base server URL for health endpoint
        server_base = self.base_url.rstrip("/")
        if server_base.endswith("/v1"):
            server_base = server_base[:-3]

        health_url = f"{server_base}/health"
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                pass
            return True
        except (urllib.error.URLError, OSError):
            return False

    def wait_until_ready(self, timeout: float = 600.0, poll_interval: float = 10.0) -> None:
        """Block until the vLLM server is ready, or raise after *timeout* seconds."""
        logger.info("Waiting for vLLM server at %s ...", self.base_url)
        elapsed = 0.0
        while elapsed < timeout:
            if self.health_check():
                logger.info("vLLM server ready after %.0fs", elapsed)
                return
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise ConnectionError(
            f"vLLM server at {self.base_url} not ready after {timeout:.0f}s"
        )

    # -- client -------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    # -- generation ---------------------------------------------------------

    def generate(self, prompt: str) -> GenerationResult:
        """Generate using vLLM's OpenAI-compatible chat endpoint.

        Retries on transient network errors (matching God's Reach robustness).
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    logprobs=True,
                    top_logprobs=5,
                    # Disable chain-of-thought thinking for Qwen3 and compatible
                    # models.  vLLM exposes this via chat_template_kwargs; other
                    # servers ignore unknown extra_body keys gracefully.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                choice = response.choices[0]
                text = choice.message.content or ""

                # Extract logprobs if available
                logprobs = None
                if choice.logprobs and choice.logprobs.content:
                    logprobs = [lp.logprob for lp in choice.logprobs.content]

                return GenerationResult(
                    text=text,
                    logprobs=logprobs,
                    finish_reason=choice.finish_reason or "",
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    },
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "vLLM request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"vLLM generation failed after {self.max_retries} attempts: {last_error}"
        ) from last_error


# ---------------------------------------------------------------------------
# Mode 2: In-process vLLM (mirrors God's Reach Qwen72BProvider)
# ---------------------------------------------------------------------------


class InProcessVLLMGenerator(BaseGenerator):
    """Load a model in-process via ``vllm.LLM`` with tensor parallelism.

    This is equivalent to God's Reach ``Qwen72BProvider`` — the model is
    loaded once into GPU memory and generation happens in the same process.
    Useful for single-shot SLURM jobs where running a separate server is
    unnecessary overhead.

    Set ``model_path`` to a **local directory** containing the pre-downloaded
    model weights (same path pattern as God's Reach).  On the cluster, set
    env vars ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        tensor_parallel_size: int = 2,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        # Resolve model path: explicit arg → env var → default HF ID
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.getenv(
                "QWEN_MODEL_PATH",
                os.getenv("QWEN72B_MODEL_PATH", "Qwen/Qwen2.5-32B-Instruct"),
            )

        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._llm = None  # vllm.LLM instance (lazy)

    # -- model validation (same checks as God's Reach) ----------------------

    @staticmethod
    def _validate_model_dir(path: str) -> None:
        """Raise early if the model directory is missing or incomplete."""
        model_dir = Path(path)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Model directory does not exist: {path}\n"
                "Download it on a login node with:\n"
                f"  huggingface-cli download <model-id> --local-dir {path}"
            )
        required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        missing = [f for f in required if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Model directory incomplete — missing: {missing}\n"
                f"Check: {path}"
            )

    # -- lazy initialization ------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the model on first use (singleton within this process)."""
        if self._llm is not None:
            return

        # Force offline mode (cluster compute nodes have no internet)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Validate path before expensive load
        if not self.model_path.startswith("http"):
            self._validate_model_dir(self.model_path)

        from vllm import LLM  # type: ignore[import-untyped]

        logger.info(
            "Loading model in-process: %s (TP=%d, max_len=%d, gpu_mem=%.2f)",
            self.model_path,
            self.tensor_parallel_size,
            self.max_model_len,
            self.gpu_memory_utilization,
        )

        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="auto",
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=False,
            disable_custom_all_reduce=True,
        )
        logger.info("Model loaded successfully: %s", self.model_path)

    # -- generation ---------------------------------------------------------

    def generate(self, prompt: str) -> GenerationResult:
        """Generate a response using in-process vLLM (matching God's Reach)."""
        self._ensure_loaded()
        assert self._llm is not None

        from vllm import SamplingParams  # type: ignore[import-untyped]

        sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0.0,
            max_tokens=self.max_tokens,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            # Qwen-specific stop tokens (from God's Reach)
            stop=["</s>", "<|endoftext|>", "<|im_end|>"],
        )

        outputs = self._llm.generate([prompt], sampling_params)

        if not outputs or len(outputs) == 0:
            raise RuntimeError("No output generated from in-process vLLM")

        output = outputs[0].outputs[0]
        return GenerationResult(
            text=output.text.strip(),
            logprobs=None,  # vLLM offline doesn't provide per-token via this API
            finish_reason=output.finish_reason or "",
        )


# ---------------------------------------------------------------------------
# Mock generator for testing
# ---------------------------------------------------------------------------


class MockGenerator(BaseGenerator):
    """Mock generator for testing without a real model."""

    def __init__(self, default_response: str = "42") -> None:
        self.default_response = default_response
        self.call_count = 0
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> GenerationResult:
        self.call_count += 1
        self.last_prompt = prompt
        return GenerationResult(text=self.default_response)
