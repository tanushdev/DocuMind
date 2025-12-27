"""
LLM integration module for DocuMind.

This module provides integration with cloud LLM services:
- Groq (fastest, recommended)
- Google Gemini
- Perplexity
- HuggingFace (fallback)
"""
from dataclasses import dataclass
from typing import Optional, AsyncGenerator
import httpx


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_time_ms: float


class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass


class GroqClient:
    """
    Client for Groq API - Ultra-fast LLM inference.
    
    Groq provides extremely fast inference for open-source models
    like Llama 3.1, Mixtral, etc. Free tier: 30 requests/min.
    
    Get your API key at: https://console.groq.com
    """
    
    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        timeout: float = 60.0
    ):
        import os
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.timeout = timeout
        
        if not self.api_key:
            raise LLMError("GROQ_API_KEY environment variable not set")
        
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> LLMResponse:
        import time
        start_time = time.perf_counter()
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            
            return LLMResponse(
                text=text,
                model=self.model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_time_ms=elapsed_ms
            )
        except httpx.HTTPError as e:
            raise LLMError(f"Groq request failed: {str(e)}")
    
    async def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 10
    
    async def close(self):
        await self._client.aclose()


class GeminiClient:
    """
    Client for Google Gemini API.
    
    Free tier: 15 requests/minute, 1500 requests/day.
    Get your API key at: https://makersuite.google.com/app/apikey
    """
    
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        timeout: float = 60.0
    ):
        import os
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.timeout = timeout
        
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY environment variable not set")
        
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> LLMResponse:
        import time
        start_time = time.perf_counter()
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            response = await self._client.post(
                url,
                json=payload,
                params={"key": self.api_key}
            )
            response.raise_for_status()
            data = response.json()
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract text from Gemini response
            text = ""
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    text = candidate["content"]["parts"][0].get("text", "")
            
            return LLMResponse(
                text=text,
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_time_ms=elapsed_ms
            )
        except httpx.HTTPError as e:
            raise LLMError(f"Gemini request failed: {str(e)}")
    
    async def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 10
    
    async def close(self):
        await self._client.aclose()


class PerplexityClient:
    """
    Client for Perplexity API.
    
    Provides access to models with real-time web search capabilities.
    Get your API key at: https://www.perplexity.ai/settings/api
    """
    
    def __init__(
        self,
        model: str = "llama-3.1-sonar-small-128k-online",
        api_key: Optional[str] = None,
        timeout: float = 60.0
    ):
        import os
        self.model = model
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.timeout = timeout
        
        if not self.api_key:
            raise LLMError("PERPLEXITY_API_KEY environment variable not set")
        
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> LLMResponse:
        import time
        start_time = time.perf_counter()
        
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            
            return LLMResponse(
                text=text,
                model=self.model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_time_ms=elapsed_ms
            )
        except httpx.HTTPError as e:
            raise LLMError(f"Perplexity request failed: {str(e)}")
    
    async def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 10
    
    async def close(self):
        await self._client.aclose()


class HuggingFaceClient:
    """
    Client for HuggingFace Inference API (fallback).
    Free tier with rate limits.
    """
    
    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        api_token: Optional[str] = None,
        timeout: float = 60.0
    ):
        import os
        self.model = model
        self.api_token = api_token or os.getenv("HF_TOKEN")
        self.timeout = timeout
        
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> LLMResponse:
        import time
        start_time = time.perf_counter()
        
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text", "")
            else:
                text = data.get("generated_text", "")
            
            return LLMResponse(
                text=text,
                model=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                total_time_ms=elapsed_ms
            )
        except httpx.HTTPError as e:
            raise LLMError(f"HuggingFace request failed: {str(e)}")
    
    async def is_available(self) -> bool:
        return True  # Always available as fallback
    
    async def close(self):
        await self._client.aclose()


class LLMServiceFactory:
    """
    Factory for LLM service instances.
    
    Priority order: Groq > Gemini > Perplexity > HuggingFace
    """
    
    _groq_instance: Optional[GroqClient] = None
    _gemini_instance: Optional[GeminiClient] = None
    _perplexity_instance: Optional[PerplexityClient] = None
    _hf_instance: Optional[HuggingFaceClient] = None
    
    @classmethod
    async def get_groq(cls) -> Optional[GroqClient]:
        """Get or create Groq client."""
        if cls._groq_instance is None:
            try:
                cls._groq_instance = GroqClient()
            except LLMError:
                return None
        return cls._groq_instance
    
    @classmethod
    async def get_gemini(cls) -> Optional[GeminiClient]:
        """Get or create Gemini client."""
        if cls._gemini_instance is None:
            try:
                cls._gemini_instance = GeminiClient()
            except LLMError:
                return None
        return cls._gemini_instance
    
    @classmethod
    async def get_perplexity(cls) -> Optional[PerplexityClient]:
        """Get or create Perplexity client."""
        if cls._perplexity_instance is None:
            try:
                cls._perplexity_instance = PerplexityClient()
            except LLMError:
                return None
        return cls._perplexity_instance
    
    @classmethod
    async def get_huggingface(cls) -> HuggingFaceClient:
        """Get or create HuggingFace client."""
        if cls._hf_instance is None:
            cls._hf_instance = HuggingFaceClient()
        return cls._hf_instance
    
    @classmethod
    async def get_available(cls):
        """
        Get the first available LLM service.
        Priority: Groq > Gemini > Perplexity > HuggingFace
        """
        # Try Groq first (fastest)
        groq = await cls.get_groq()
        if groq and await groq.is_available():
            return groq
        
        # Try Gemini
        gemini = await cls.get_gemini()
        if gemini and await gemini.is_available():
            return gemini
        
        # Try Perplexity
        perplexity = await cls.get_perplexity()
        if perplexity and await perplexity.is_available():
            return perplexity
        
        # Fall back to HuggingFace
        return await cls.get_huggingface()
