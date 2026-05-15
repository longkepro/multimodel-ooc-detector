import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ============== API KEYS ==============
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    # ============== ENVIRONMENT DETECTION ==============
    # Không dùng IS_KAGGLE để ép mode nữa. Sử dụng biến USE_VLLM do bạn set ở Cell 3.
    # Nếu "True" -> Dùng GPU của Kaggle tải vLLM. Nếu "False" -> Dùng Groq API cho nhanh.
    USE_VLLM: bool = os.getenv("USE_VLLM", "False").lower() == "false"
    
    # Biến này chỉ dùng để biết có đang ở Kaggle không (dành cho các tác vụ lưu cache)
    IS_KAGGLE: bool = os.path.exists("/kaggle/input")

    # ============== MODEL CONFIG ==============
    if USE_VLLM:
        # Local GPU (Kaggle T4/P100): Tải trực tiếp bản 8B từ Hugging Face
        # Không dùng đường dẫn "/kaggle/input/..." cứng nhắc nữa
        # MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        MODEL_PATH: str = MODEL_NAME # Cho vLLM tự động tải weights từ internet về
        BASE_URL: None = None
        # Tensor parallel: T4 x2 → size=2, P100 x1 → size=1
        TENSOR_PARALLEL_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "2"))
    else:
        # API Mode (Local hoặc test Kaggle bằng API)
        MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.1-70b-versatile") # Tên trên Groq
        MODEL_PATH: str = ""
        BASE_URL: str = "https://api.groq.com/openai/v1"
        TENSOR_PARALLEL_SIZE: int = 1

    # ============== PIPELINE CONFIG ==============
    MAX_EVIDENCE: int = int(os.getenv("MAX_EVIDENCE", "3"))
    MAX_REFINE: int = int(os.getenv("MAX_REFINE", "3"))
    REFINE_THRESHOLD: float = float(os.getenv("REFINE_THRESHOLD", "0.75"))

    @classmethod
    def log_env(cls) -> None:
        """Gọi 1 lần duy nhất ở entry point."""
        if cls.USE_VLLM:
            print(f"🚀 LOCAL GPU DETECTED → vLLM | model={cls.MODEL_NAME} | tp={cls.TENSOR_PARALLEL_SIZE}")
        else:
            print(f"💻 API MODE DETECTED → Groq API | model={cls.MODEL_NAME}")

    @classmethod
    def validate(cls) -> None:
        """Kiểm tra config hợp lệ trước khi chạy pipeline."""
        errors = []
        if not cls.USE_VLLM and not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is missing (cần cho API Mode)")
        if not cls.SERPAPI_API_KEY:
            errors.append("SERPAPI_API_KEY is missing in .env")
        if errors:
            raise EnvironmentError("Config errors:\n" + "\n".join(f"  - {e}" for e in errors))