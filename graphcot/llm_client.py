"""
graphcot: Standalone LLM Client Module
------------------------------------------
Provides a simple wrapper for OpenAI/vLLM compatible API calls.
"""

import aiohttp
import asyncio
import json
import logging
import random
from typing import Optional, Dict, Any, List

from graphcot import BaseLLMClient

logger = logging.getLogger("LLMClient")


class LLMClientError(Exception):
    """Raised when LLM API calls fail after all retries."""
    pass


class LLMClient(BaseLLMClient):
    """
    A standalone async HTTP client for OpenAI-compatible APIs.
    
    Supports async context manager protocol for proper resource cleanup::
    
        async with LLMClient(base_url="...") as client:
            answer = await client.generate_answer("Hello")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            connector = aiohttp.TCPConnector(ssl=False)
            self._session = aiohttp.ClientSession(headers=headers, trust_env=True, connector=connector)
        return self._session

    async def close(self):
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def generate_answer(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate a response for the given prompt using the chat completion endpoint.
        
        Args:
            prompt: The input prompt text.
            temperature: Sampling temperature.
            
        Returns:
            The generated text response.
            
        Raises:
            LLMClientError: If all retry attempts fail.
        """
        session = await self._get_session()
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2048
        }
        
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=payload, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(f"API Error {resp.status}: {error_text}. Attempt {attempt+1}/{self.max_retries}")
                        last_error = LLMClientError(f"API returned status {resp.status}: {error_text}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    
                    data = await resp.json()
                    
                    # Handle non-standard error responses (e.g. rate limiting)
                    # Some APIs return HTTP 200 but with error payload like:
                    # {"status": "449", "msg": "rate limit exceeded", "body": null}
                    if "choices" not in data:
                        error_msg = data.get("msg", data.get("message", str(data)))
                        api_status = data.get("status", "unknown")
                        logger.warning(
                            f"API non-standard response (status={api_status}): {error_msg}. "
                            f"Attempt {attempt+1}/{self.max_retries}"
                        )
                        last_error = LLMClientError(f"API error (status={api_status}): {error_msg}")
                        # Rate limit: wait longer with jitter
                        wait = (2 ** attempt) + random.uniform(0.5, 2.0)
                        await asyncio.sleep(wait)
                        continue
                    
                    return data["choices"][0]["message"]["content"]
            except LLMClientError:
                raise
            except asyncio.TimeoutError as e:
                logger.error(f"Request timeout (attempt {attempt+1}/{self.max_retries}): The API took longer than {self.timeout}s to respond.")
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Request failed (attempt {attempt+1}/{self.max_retries}): {type(e).__name__} - {str(e)}")
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                
        raise LLMClientError(
            f"All {self.max_retries} attempts failed. Last error: {last_error}"
        )
