import os
import time
from typing import List, Dict, Any, Optional

class LLMProvider:
    def __init__(self):
        self.model = os.environ.get("MODEL", "llama-3.3-70b-versatile")
        self.provider = os.environ.get("LLM_PROVIDER", "groq").lower()
        
        # Initialize clients lazily
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_client = None
        self._groq_client = None

    def _get_groq_client(self):
        if self._groq_client is None:
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            api_key = os.environ.get("OPENAI_API_KEY", "ollama")
            self._openai_client = OpenAI(base_url=base_url, api_key=api_key)
        return self._openai_client

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def _get_gemini_client(self):
        if self._gemini_client is None:
            import google.generativeai as genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            self._gemini_client = genai
        return self._gemini_client

    def call(self, messages: List[Dict[str, str]], system_prompt: str, max_tokens: int = 1500, temperature: float = 0.0) -> str:
        if self.provider == "groq":
            return self._call_groq(messages, system_prompt, max_tokens, temperature)
        elif self.provider == "ollama" or self.provider == "openai":
            return self._call_openai(messages, system_prompt, max_tokens, temperature)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, system_prompt, max_tokens, temperature)
        elif self.provider == "gemini":
            return self._call_gemini(messages, system_prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_groq(self, messages, system_prompt, max_tokens, temperature):
        client = self._get_groq_client()
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        # Remove groq/ prefix if present
        model_name = self.model
        if model_name.startswith("groq/"):
            model_name = model_name[5:]
            
        resp = client.chat.completions.create(
            model=model_name,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def _call_openai(self, messages, system_prompt, max_tokens, temperature):
        client = self._get_openai_client()
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        resp = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def _call_anthropic(self, messages, system_prompt, max_tokens, temperature):
        client = self._get_anthropic_client()
        # Anthropic uses a separate system parameter
        resp = client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.content[0].text

    def _call_gemini(self, messages, system_prompt, max_tokens, temperature):
        genai = self._get_gemini_client()
        model = genai.GenerativeModel(self.model, system_instruction=system_prompt)
        
        # Convert messages to Gemini format
        history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})
        
        chat = model.start_chat(history=history)
        last_msg = messages[-1]["content"]
        resp = chat.send_message(last_msg, generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ))
        return resp.text

def get_llm():
    return LLMProvider()
