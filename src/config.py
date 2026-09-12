from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    data_path: str = os.getenv(
        "DATA_PATH",
        "data/raw/customer_support_on_twitter.csv"
    )
    golden_path: str = os.getenv(
        "GOLDEN_PATH",
        "data/golden_set.csv"
    )
    output_dir: str = os.getenv(
        "OUTPUT_DIR",
        "outputs"
    )
    brand: str = os.getenv(
        "BRAND",
        "AppleSupport"
    )
    openai_api_key: str = os.getenv(
        "OPENAI_API_KEY",
        ""
    )
    openai_model: str = os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    )


settings = Settings()

Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
