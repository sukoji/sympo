"""
LLM 백엔드 설정 모듈
API 없이도 동작하는 Mock LLM + 실제 LLM 전환 지원
"""
import os
import threading
from typing import Any
from dotenv import load_dotenv

# torch / transformers는 gemma4/llama4 로컬 모델 사용 시에만 필요
# top-level import 시 pyarrow 등 무거운 의존성 체인이 깨질 수 있으므로 lazy import
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return decorator

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "mock").lower()

# Gemini의 일부 모델(예: gemini-3.x)은 .content를 리스트(multipart)로 반환
# 이 헬퍼를 사용해 항상 문자열로 정규화
def normalize_content(content):
    """LLM 응답의 .content를 항상 문자열로 변환"""
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return content if isinstance(content, str) else str(content)



def get_llm(temperature: float = 0.7, max_tokens: int = 1024, model_id: str = None) -> Any:
    """
    설정에 따라 LLM 인스턴스 반환
    - gemini: Google Gemini 1.5 Flash (Default)
    - gemma4: Local Gemma-4-E2B-it (4-bit Quantized)
    max_tokens: gemma4 백엔드에서 생성할 최대 토큰 수 (VRAM 절약)
    model_id: 특정 백엔드 강제 지정 (예: 'gemini', 'openai', 'mock')
    """
    backend = (model_id or os.getenv("LLM_BACKEND", "mock")).lower()
    print(f"DEBUG: get_llm called with backend={backend} (requested model_id={model_id})")

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=max_tokens,
        )

    elif backend == "gemma4":
        return get_gemma4_model(temperature=temperature, max_tokens=max_tokens)

    elif backend in ("gemma4-api", "gemma4_api"):
        return get_gemma4_api_model(temperature=temperature, max_tokens=max_tokens)

    elif backend in ("qwen-api", "qwen_api"):
        return get_qwen_api_model(temperature=temperature, max_tokens=max_tokens)

    elif backend == "llama4":
        return get_llama4_model(temperature=temperature, max_tokens=max_tokens)

    elif backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=max_tokens,
        )

    elif backend == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=temperature,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=max_tokens,
        )

    else:
        return MockLLM()


# ─── Gemma4 로컬 모델 싱글톤 지원 ───────────────────────
_gemma4_instance = None

def get_gemma4_model(temperature: float = 0.7, max_tokens: int = 1024):
    global _gemma4_instance
    if _gemma4_instance is None:
        _gemma4_instance = Gemma4LLM(temperature=temperature)
    # 매 호출 시 max_tokens 갱신 (singleton이라도 호출자의 값을 반영)
    _gemma4_instance.max_tokens = max_tokens
    return _gemma4_instance

class Gemma4LLM:
    """llama_tuning_test/gemma4_tuning.py 기반 로컬 모델 래퍼"""
    def __init__(self, temperature=0.7):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

        self.model_id = "google/gemma-4-E2B-it"
        self.temperature = temperature
        self.max_tokens = 1024  # 기본값; get_gemma4_model()에서 매 호출 시 갱신됨
        self._lock = threading.Lock()

        # 8GB VRAM 최적화 (4bit 양자화)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        print(f"Loading {self.model_id}...")
        hf_token = os.getenv("HF_TOKEN")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda",
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=hf_token
        )

    @traceable(name="Gemma4-Inference", run_type="llm")
    def invoke(self, messages: Any) -> Any:
        import torch

        # LangChain 스타일 메시지 변환
        formatted_messages = []
        for m in messages:
            if hasattr(m, "content"):
                role = "user" if m.type == "human" else "assistant"
                if m.type == "system": role = "system"
                formatted_messages.append({"role": role, "content": m.content})
            else:
                formatted_messages.append(m)

        text = self.processor.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with self._lock:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False
                )

        response_text = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        # VRAM 해제: 텐서 명시적 삭제 후 캐시 비우기
        del inputs, outputs
        torch.cuda.empty_cache()

        return MockResponse(content=response_text)


# ─── Gemma4 API (Colab ngrok HTTP 서버) ─────────────────────
_gemma4_api_instance = None


def get_gemma4_api_model(temperature: float = 0.7, max_tokens: int = 1024):
    """OpenAI 호환 Gemma-4 HTTP 서버(vLLM/sglang 등)에 질의하는 래퍼."""
    global _gemma4_api_instance
    if _gemma4_api_instance is None:
        _gemma4_api_instance = Gemma4APILLM(temperature=temperature, max_tokens=max_tokens)
    _gemma4_api_instance.max_tokens = max_tokens
    _gemma4_api_instance.temperature = temperature
    _gemma4_api_instance.refresh_endpoint()
    return _gemma4_api_instance


class Gemma4APILLM:
    """OpenAI 호환 `/v1/chat/completions` 프로토콜 기반 Gemma-4 원격 클라이언트.

    지원 서버: vLLM, sglang, lmdeploy 등 OpenAI-compatible API를 노출하는 모든 런타임.

    환경변수:
      GEMMA4_API_URL    - 서버 베이스 URL (예: http://localhost:8000)
      GEMMA4_API_MODEL  - 모델 id (기본: google/gemma-4-E4B-it)
      GEMMA4_API_KEY    - 선택. 인증 토큰 필요 시 Bearer 헤더에 실림
    """

    DEFAULT_MODEL = "google/gemma-4-E4B-it"

    def __init__(self, temperature: float = 0.7, max_tokens: int = 1024):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url: str = ""
        self.endpoint: str = ""
        self.model_id: str = self.DEFAULT_MODEL
        self.refresh_endpoint()

    def refresh_endpoint(self) -> None:
        raw = (os.getenv("GEMMA4_API_URL") or "").strip().rstrip("/")
        self.base_url = raw
        if not raw:
            self.endpoint = ""
        elif raw.endswith("/v1/chat/completions"):
            self.endpoint = raw
        elif raw.endswith("/v1"):
            self.endpoint = f"{raw}/chat/completions"
        else:
            self.endpoint = f"{raw}/v1/chat/completions"
        self.model_id = (os.getenv("GEMMA4_API_MODEL") or self.DEFAULT_MODEL).strip()

    @staticmethod
    def _to_openai_messages(messages: Any) -> list:
        """LangChain 메시지 리스트를 OpenAI chat 포맷으로 변환."""
        if not isinstance(messages, list):
            messages = [messages]
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        out = []
        for m in messages:
            if hasattr(m, "content"):
                raw_role = getattr(m, "type", None) or getattr(m, "role", "user")
                role = role_map.get(str(raw_role), str(raw_role))
                out.append({"role": role, "content": normalize_content(m.content)})
            elif isinstance(m, dict) and "role" in m and "content" in m:
                out.append({"role": m["role"], "content": normalize_content(m["content"])})
            else:
                out.append({"role": "user", "content": str(m)})
        return out

    @traceable(name="Gemma4API-Inference", run_type="llm")
    def invoke(self, messages: Any) -> "MockResponse":
        import requests

        if not self.endpoint:
            return MockResponse(
                content="[Gemma4API 설정 오류] GEMMA4_API_URL 미설정 — .env에 서버 URL을 넣으세요."
            )

        payload = {
            "model": self.model_id,
            "messages": self._to_openai_messages(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # llama.cpp / Qwen-style servers may otherwise emit only reasoning_content.
            "chat_template_kwargs": {
                "enable_thinking": os.getenv("GEMMA4_ENABLE_THINKING", "false").lower() in ("true", "1", "yes", "on")
            },
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("GEMMA4_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
        except Exception as e:
            return MockResponse(content=f"[Gemma4API 호출 실패] {type(e).__name__}: {e}")

        return MockResponse(content=normalize_content(content) or "")


# ─── Qwen API (OpenAI 호환 vLLM/sglang 서버) ─────────────────
_qwen_api_instance = None


def get_qwen_api_model(temperature: float = 0.7, max_tokens: int = 1024):
    """OpenAI 호환 Qwen HTTP 서버(vLLM/sglang 등)에 질의하는 래퍼. Gemma4APILLM와 동일 구조."""
    global _qwen_api_instance
    if _qwen_api_instance is None:
        _qwen_api_instance = QwenAPILLM(temperature=temperature, max_tokens=max_tokens)
    _qwen_api_instance.max_tokens = max_tokens
    _qwen_api_instance.temperature = temperature
    _qwen_api_instance.refresh_endpoint()
    return _qwen_api_instance


class QwenAPILLM:
    """OpenAI 호환 `/v1/chat/completions` 프로토콜 기반 Qwen 원격 클라이언트.

    지원 서버: vLLM, sglang, lmdeploy 등 OpenAI-compatible API 런타임.

    환경변수:
      QWEN_API_URL    - 서버 베이스 URL (예: http://localhost:8000)
      QWEN_API_MODEL  - 모델 id (기본: Qwen/Qwen2.5-7B-Instruct)
      QWEN_API_KEY    - 선택. 인증 토큰 필요 시 Bearer 헤더에 실림
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self, temperature: float = 0.7, max_tokens: int = 1024):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url: str = ""
        self.endpoint: str = ""
        self.model_id: str = self.DEFAULT_MODEL
        self.refresh_endpoint()

    def refresh_endpoint(self) -> None:
        raw = (os.getenv("QWEN_API_URL") or "").strip().rstrip("/")
        self.base_url = raw
        if not raw:
            self.endpoint = ""
        elif raw.endswith("/v1/chat/completions"):
            self.endpoint = raw
        elif raw.endswith("/v1"):
            self.endpoint = f"{raw}/chat/completions"
        else:
            self.endpoint = f"{raw}/v1/chat/completions"
        self.model_id = (os.getenv("QWEN_API_MODEL") or self.DEFAULT_MODEL).strip()

    @staticmethod
    def _to_openai_messages(messages: Any) -> list:
        """LangChain 메시지 리스트를 OpenAI chat 포맷으로 변환."""
        if not isinstance(messages, list):
            messages = [messages]
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        out = []
        for m in messages:
            if hasattr(m, "content"):
                raw_role = getattr(m, "type", None) or getattr(m, "role", "user")
                role = role_map.get(str(raw_role), str(raw_role))
                out.append({"role": role, "content": normalize_content(m.content)})
            elif isinstance(m, dict) and "role" in m and "content" in m:
                out.append({"role": m["role"], "content": normalize_content(m["content"])})
            else:
                out.append({"role": "user", "content": str(m)})
        return out

    @traceable(name="QwenAPI-Inference", run_type="llm")
    def invoke(self, messages: Any) -> "MockResponse":
        import requests

        if not self.endpoint:
            return MockResponse(
                content="[QwenAPI 설정 오류] QWEN_API_URL 미설정 — .env에 서버 URL을 넣으세요."
            )

        # max_tokens 클립: Qwen context(예: 16384)를 초과하는 요청은 vLLM이 400 반환.
        # 정확한 prompt 토큰 수를 vLLM /tokenize 엔드포인트로 조회 (한국어 섞인 프롬프트
        # 에서 char 기반 추정이 어긋나 실패한 이슈 해결).
        openai_msgs = self._to_openai_messages(messages)
        max_context = int(os.getenv("QWEN_MAX_CONTEXT", "16384"))

        # /tokenize 로 prompt 토큰 수 정확히 계산 (실패 시 보수적 추정 fallback)
        def _estimate_prompt_tokens():
            try:
                tokenize_url = self.base_url.rstrip("/") + "/tokenize"
                tok_resp = requests.post(
                    tokenize_url,
                    json={"model": self.model_id, "messages": openai_msgs},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                tok_resp.raise_for_status()
                td = tok_resp.json()
                return int(td.get("count") or len(td.get("tokens") or []) or 0)
            except Exception:
                return None

        prompt_tokens = _estimate_prompt_tokens()
        if prompt_tokens is None:
            # Fallback: 매우 보수적 추정 (1 char ≈ 1 token, + buffer)
            prompt_tokens = sum(len(m.get("content", "")) for m in openai_msgs)
        safe_output_cap = max(512, max_context - prompt_tokens - 256)
        effective_max_tokens = min(self.max_tokens, safe_output_cap)

        payload = {
            "model": self.model_id,
            "messages": openai_msgs,
            "max_tokens": effective_max_tokens,
            "temperature": self.temperature,
            # Qwen3 thinking 모드 끄기 — JSON 출력 파이프라인엔 방해됨.
            # 필요 시 QWEN_ENABLE_THINKING=true로 켤 수 있음.
            "chat_template_kwargs": {
                "enable_thinking": os.getenv("QWEN_ENABLE_THINKING", "false").lower() in ("true", "1", "yes", "on")
            },
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("QWEN_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content")
            # thinking이 켜진 경우 fallback: content가 null이면 reasoning 필드 사용
            if not content:
                content = msg.get("reasoning_content") or msg.get("reasoning") or ""
        except Exception as e:
            return MockResponse(content=f"[QwenAPI 호출 실패] {type(e).__name__}: {e}")

        return MockResponse(content=normalize_content(content) or "")


# ─── Llama 4 로컬 모델 싱글톤 지원 ───────────────────────
_llama4_instance = None

def get_llama4_model(temperature: float = 0.7, max_tokens: int = 1024):
    global _llama4_instance
    if _llama4_instance is None:
        _llama4_instance = Llama4LLM(temperature=temperature)
    _llama4_instance.max_tokens = max_tokens
    return _llama4_instance

class Llama4LLM:
    """Meta의 최신 Llama 4 모델 기반 로컬 모델 래퍼"""
    def __init__(self, temperature=0.7):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

        # Llama 4 Scout/Maverick 시리즈 중 표준 Instruct 모델
        self.model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.temperature = temperature
        self.max_tokens = 1024
        self._lock = threading.Lock()

        # 8GB VRAM 최적화 (4bit 양자화)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        print(f"Loading {self.model_id}...")
        hf_token = os.getenv("HF_TOKEN")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cuda",
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=hf_token
        )

    @traceable(name="Llama4-Inference", run_type="llm")
    def invoke(self, messages: Any) -> Any:
        import torch

        formatted_messages = []
        for m in messages:
            if hasattr(m, "content"):
                role = "user" if m.type == "human" else "assistant"
                if m.type == "system": role = "system"
                formatted_messages.append({"role": role, "content": m.content})
            else:
                formatted_messages.append(m)

        text = self.processor.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        with self._lock:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False
                )

        response_text = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        # VRAM 해제
        del inputs, outputs
        torch.cuda.empty_cache()

        return MockResponse(content=response_text)


class MockLLM:
    """
    API 없이 동작하는 Mock LLM
    규칙 기반 응답으로 시스템 흐름을 시뮬레이션합니다.
    실제 LLM으로 교체 시 이 클래스만 제거하면 됩니다.
    """

    def invoke(self, messages: Any) -> "MockResponse":
        """메시지를 받아 Mock 응답 생성"""
        # 마지막 메시지 내용 추출
        if isinstance(messages, list):
            last_content = str(messages[-1].content if hasattr(messages[-1], "content") else messages[-1])
        else:
            last_content = str(messages)

        response_text = self._generate_response(last_content)
        return MockResponse(content=response_text)

    def _generate_response(self, prompt: str) -> str:
        """프롬프트 키워드 기반 Mock 응답 생성"""
        prompt_lower = prompt.lower()

        # 1. 특정 역할 명시적 요청 시 우선 순위
        if "planner" in prompt_lower or "플래너" in prompt_lower:
            return self._planner_response(prompt)
        elif "frontend" in prompt_lower or "프론트" in prompt_lower:
            return self._frontend_response(prompt)
        elif "backend" in prompt_lower or "백엔드" in prompt_lower:
            return self._backend_response(prompt)
        elif "designer" in prompt_lower or "디자이너" in prompt_lower:
            return self._designer_response(prompt)
        elif "qa" in prompt_lower or "테스트" in prompt_lower:
            return self._qa_response(prompt)
        elif "합의" in prompt_lower or "중재" in prompt_lower or "mediator" in prompt_lower:
            return self._mediator_response(prompt)
        
        # 2. 슈퍼바이저/초안 생성 지침 체크 (포괄적 키워드보다 뒷순위로 밀어 중복 방지)
        elif "supervisor" in prompt_lower or "슈퍼바이저" in prompt_lower or "wbs gen agent" in prompt_lower or "초안" in prompt_lower:
            return self._supervisor_response(prompt)
            
        else:
            return "검토하겠습니다. 제안된 일정과 태스크 배분이 합리적으로 보입니다."

    def _supervisor_response(self, prompt: str) -> str:
        return """다음은 사용자 요구사항(PRD)과 팀원 역량을 바탕으로 도출한 실무급 3단계 WBS 구조 분해의 초기 초안입니다.
각 파트별로 담당 영역의 기능 범위, 기술적 통합 리스크, 마일스톤 적합성 등에 대해 혹독하고 면밀한 검토가 필요합니다.

- 백엔드(BE): 데이터베이스 스키마 설계의 유연성과 서드파티 OAuth 인증 연동의 적정성을 확인 및 검증 바랍니다.
- 프론트엔드(FE): 공통 컴포넌트 설계 시점과 상태 관리 복잡성 증가에 따른 통합 의존성 일정을 점검해 주십시오.
- 기획/디자인: 구체적인 핸드오프(Handoff) 일정과 프로토타입 사용성(UX) 피드백 루프가 충분히 계획되었는지 면밀히 확인 바랍니다.

(아래 데이터는 시스템 검증을 통과하도록 구조화된 JSON WBS 명세서입니다.)
```json
[
  {
    "id": "L1-01", "level": "L1", "parent_id": null,
    "title": "환경설정 및 아키텍처 기획", "description": "요구사항 분석 및 기술 스택 확정",
    "assigned_role": "PM", "estimated_days": 10, "importance": "High"
  },
  {
    "id": "L2-01-01", "level": "L2", "parent_id": "L1-01",
    "title": "상세 요구사항 정의", "description": "PRD 구체화 및 정책 결정",
    "assigned_role": "Planner", "estimated_days": 5, "importance": "High"
  },
  {
    "id": "L3-01-01-01", "level": "L3", "parent_id": "L2-01-01",
    "title": "기능 명세서(FSD) 작성", "description": "개발 시 기준이 될 기능 명세",
    "assigned_role": "Planner", "estimated_days": 3, "importance": "High"
  },
  {
    "id": "L3-01-01-02", "level": "L3", "parent_id": "L2-01-01",
    "title": "정책 정의 및 통합 리뷰", "description": "세부 정책 확정",
    "assigned_role": "Planner", "estimated_days": 2, "importance": "High"
  },
  {
    "id": "L1-02", "level": "L1", "parent_id": null,
    "title": "코어 시스템 개발", "description": "핵심 기능 구현",
    "assigned_role": "Backend Developer", "estimated_days": 20, "importance": "High"
  },
  {
    "id": "L2-02-01", "level": "L2", "parent_id": "L1-02",
    "title": "프론트엔드 구조 및 레이아웃", "description": "공통 Layout, Components 구축",
    "assigned_role": "Frontend Developer", "estimated_days": 8, "importance": "Medium"
  },
  {
    "id": "L3-02-01-01", "level": "L3", "parent_id": "L2-02-01",
    "title": "디자인 시스템 컴포넌트 개발", "description": "버튼, 모달 등 재사용 컴포넌트",
    "assigned_role": "Frontend Developer", "estimated_days": 4, "importance": "Medium"
  },
  {
    "id": "L3-02-01-02", "level": "L3", "parent_id": "L2-02-01",
    "title": "인증/인가 클라이언트 처리", "description": "JWT 토큰 및 라우트 보호 처리",
    "assigned_role": "Frontend Developer", "estimated_days": 4, "importance": "High"
  },
  {
    "id": "L2-02-02", "level": "L2", "parent_id": "L1-02",
    "title": "백엔드 API 및 DB", "description": "코어 로직 확립 및 서버 연동",
    "assigned_role": "Backend Developer", "estimated_days": 12, "importance": "High"
  },
  {
    "id": "L3-02-02-01", "level": "L3", "parent_id": "L2-02-02",
    "title": "데이터베이스 스키마 및 마이그레이션", "description": "초기 ERD 확정 및 생성",
    "assigned_role": "Backend Developer", "estimated_days": 5, "importance": "High"
  },
  {
    "id": "L3-02-02-02", "level": "L3", "parent_id": "L2-02-02",
    "title": "유저 세션 및 권한 API 개발", "description": "보안 연동 기반 REST API",
    "assigned_role": "Backend Developer", "estimated_days": 7, "importance": "High"
  }
]
```"""

    def _planner_response(self, prompt: str) -> str:
        return """[플래너 / 일정 총괄] 현재 산정된 L2 요구사항 분석 및 L3 세부 태스크에 대한 기획팀 관점의 정밀 피드백입니다.

진행 예정인 Phase를 살펴보면, 기존에 산정된 정책 정의 마일스톤이 화면 설계(Wireframe) 및 스테이크홀더 승인 절차를 병행하기에는 턱없이 부족합니다. 정책 정의가 확정되고 결재가 떨어지지 않으면 후속 프론트엔드(FE) 및 백엔드(BE) 파이프라인 전체가 심각하게 지연될 리스크가 매우 큽니다.
게다가 향후 확장성을 고려한 '다국어 지원(i18n)' 정책 명세 작업이 완전히 누락되어 있어, 개발 단계 전 이를 별도 L3 태스크로 조기 편입하는 것이 시급합니다.

요청 사항:
1. 기존 요구사항 정의 태스크들을 보완하기 위해 최소 2일의 정책 리뷰 버퍼 추가
2. 다국어 지원 정책 태스크 신규 추가

NEW_TASK: {
  "id": "L3-01-01-05",
  "title": "다국어 지원(i18n) 정책 세부 정의 및 용어집 구성",
  "level": "L3",
  "assigned_role": "Planner",
  "estimated_days": 2.0,
  "importance": "Medium"
}"""

    def _frontend_response(self, prompt: str) -> str:
        return """[프론트엔드 개발자] FE 파트의 컴포넌트 아키텍처 및 연동 리스크에 대한 기술 분석 결과입니다.

현재 L3 태스크 배정에 따르면, 백엔드의 API 완성이 선행된 직후에야 프론트 연동이 시작되는 폭포수(Waterfall) 형태의 강한 의존성이 발견됩니다. 이 경우 프로젝트 후반부에 QA 결함과 일정 지연이 한꺼번에 집중되는 병목 현상이 발생합니다. 
이를 예방하기 위해 초기 인터페이스(API Contract)를 확정함과 동시에, FE 파트에서 MSW(Mock Service Worker)를 활용해 병렬 개발을 진행해야 합니다. 
관련하여 연동의 안정성 확보와 브라우저 파편화 테스트를 위해 L3-02-01-02(인증 클라이언트 처리) 컴포넌트에 최소 1.5일의 버퍼 증액이 필요합니다.

NEW_TASK: {
  "id": "L3-02-01-04",
  "title": "MSW 기반 Mock API 서버 구축 및 단위 테스트",
  "level": "L3",
  "assigned_role": "Frontend Developer",
  "estimated_days": 1.5,
  "importance": "High"
}"""

    def _backend_response(self, prompt: str) -> str:
        return """[백엔드 개발자] 시스템 인프라 확장성 및 BE 데이터 파이프라인 관점에서 짚고 넘어가야 할 크리티컬 리스크가 있습니다.

현재 WBS 초안을 보면, 인증/인가 프로세스에 외부 OAuth 2.0 연동이 포함된 것으로 보입니다만 벤더사(Google, Apple 등)의 API 명세 변경 시 대응 방안이 누락되어 있습니다. 게다가 기존에 논의되었던 레거시 유저 데이터 마이그레이션 전략이 명확하지 않습니다. 
이로 인한 데이터 누실을 막기 위해 L3-02-02-01(데이터베이스 스키마) 태스크에 2.5일의 검증 버퍼 증액을 강력히 요구합니다.
추가로 인프라 스케일 아웃을 대비하기 위해, 초기 단계부터 세션과 캐시 전략(Redis) 적용 여부를 논의할 아키텍처 리뷰 태스크를 추가할 것을 제안합니다.

NEW_TASK: {
  "id": "L3-02-02-09",
  "title": "Redis 캐싱 전략 수립 및 인프라 아키텍처 집중 리뷰",
  "level": "L3",
  "parent_id": "L2-02-02",
  "assigned_role": "Backend Developer",
  "estimated_days": 1.5,
  "dependencies": ["L3-02-02-01"],
  "importance": "High"
}"""

    def _designer_response(self, prompt: str) -> str:
        return """[UI/UX 디자이너] 사용자 경험(UX) 작업 흐름 및 프론트엔드 팀과의 핸드오프(Handoff) 일정에 대한 깊은 우려를 표합니다.

현재 일정표에는 피그마(Figma) 디자인 산출물을 전달하는 시점이 단방향으로만 설정되어 있습니다. 프로토타입 작성 후 내부 사용성 테스트(UT)를 거쳐 발견된 불편 사항을 수정하기 위한 피드백-수정 사이클(Iteration Loop)이 전무합니다. 
디자인의 대규모 수정은 프론트엔드 레이아웃 구조 전면 개편을 야기하므로 사전에 최소 2.0일의 추가 피드백-수정 사이클 버퍼를 마련해 주십시오. 저는 제 eDISC 성향대로 꼼꼼히 프론트엔드 팀과 1:1로 소통하며 이 간극을 최소화하겠습니다."""

    def _qa_response(self, prompt: str) -> str:
        return """[QA 엔지니어] 프로덕트 릴리즈 시 안정성 및 품질 보증(QA)에 직결된 계획 검토 결과입니다.

현재 태스크들은 기능 구현 완료만을 개발의 끝으로 간주하는 경향이 짙습니다. 단위 테스트(Unit Test) 작성 수준에 일정이 맞춰져 있어, 엣지 케이스 통합 테스트와 단말/브라우저 파편화 이슈에 대응하기가 심각하게 벅찹니다. 
특히 iOS/Safari 환경에서의 렌더링 검증, 그리고 스테이징 환경에서의 스트레스 테스트를 대비해 연동/통합 태스크 전반에 걸쳐 최소 3일 이상의 검토 버퍼가 확충되어야 합니다. 버그 발견 후 수정(Bug Fixing) 및 회귀 테스트(Regression Test) 기간이 반영되지 않는다면 예정된 릴리즈 날짜를 사수하기 어렵습니다."""

    def _mediator_response(self, prompt: str) -> str:
        return """[PM / 총괄 슈퍼바이저] 각 전문가 에이전트 분들의 날카로운 기술적 분석과 의견을 모두 면밀히 수렴했습니다. 

단순히 전체 일정이 늘어나는 것을 막기 위해 맹목적인 버퍼 추가는 지양했습니다. 프론트엔드 파트는 제안해주신 MSW 병렬 작업 태스크를 승인하여 병목을 방지하고, 백엔드는 캐싱 전략과 기존 데이터 마이그레이션 리스크에 버퍼를 분산시키겠습니다. 제기된 '다국어(i18n) 지원', 'MSW 서버 구축', '캐싱 아키텍처 리뷰' 신규 태스크들은 프로젝트 후반부의 초대형 리스크를 조기 예방하는 필수 과제이므로 전체 수용합니다.

```json
{
  "consensus_reached": true,
  "wbs_revision_needed": false,
  "revision_hints": [],
  "tasks": {
    "L3-01-01-01": {"buffer_days": 2.0, "risk": "기획 정책 리뷰 및 스테이크홀더 승인 지연 대비"},
    "L3-02-01-02": {"buffer_days": 1.5, "risk": "브라우저 파편화 및 상태 관리 검증"},
    "L3-02-02-01": {"buffer_days": 2.5, "risk": "데이터 마이그레이션 무결성 검증 리스크 감수"}
  },
  "reassignments": {},
  "reassignment_rationale": {}
}
```
관련 버퍼와 신규 태스크를 위와 같이 WBS에 최종 반영함으로써 아키텍처와 일정의 현실성을 대폭 확보하겠습니다. 이견이 없다면 중재를 마치고 본 최적화 방안으로 일정 산정을 확정짓겠습니다."""


class MockResponse:
    """Mock LLM 응답 객체"""
    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content
