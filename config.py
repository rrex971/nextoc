import os
from dotenv import load_dotenv

load_dotenv()

# api keys — loaded from environment
# empty string default allows imports without groww api key configured
GROWW_API_KEY = os.environ.get("GROWW_API_KEY", "")
GROWW_API_SECRET = os.environ.get("GROWW_API_SECRET", "")
GROWW_AUTH_MODE = os.environ.get("GROWW_AUTH_MODE", "key_secret")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")

# ranking / gamma
LIQUIDITY_FLOOR = 1_000_000
HURDLE = 0.002
TOP_K = 10

# kelly
KELLY_MULTIPLIER = 0.25
MAX_POSITION_PCT = 0.20
WIN_PROBABILITY_DEFAULT = 0.55

# llm judge
LAMBDA = 0.30
LLM_MAX_TOKENS = 200
LLM_RETRY_ATTEMPTS = 1

# news
NEWS_MAX_AGE_HOURS = 48
NEWS_TOP_N = 3
NEWS_DOMAINS = [
    "economictimes.indiatimes.com",
    "moneycontrol.com/news",
    "livemint.com",
    "business-standard.com",
    "financialexpress.com",
]

# feature engineering
MIN_LOOKBACK_DAYS = 60
FETCH_DAYS = 180  # groww daily candle endpoint 180-day hard limit
TARGET_RETURN_CLIP = 0.10
OUTLIER_SIGMA = 3.0

# regime
REGIME_WINDOW = 60
REGIME_Z_THRESHOLD = -2.5

# model / training
TFT_HIDDEN_DIM = 64
TRAINING_LOOKBACK_DAYS = 252
ADAPTER_DEFAULT_LR = 1e-3
ADAPTER_PENALTY_LR = 5e-3
ADAPTER_FINETUNE_DAYS = 5
WIN_PROBABILITY_MIN_TRADES = 30

# paper trading
STARTING_PORTFOLIO = 100_000
MAX_OPEN_POSITIONS = 5

# paths
DB_PATH = "data/nextoc.db"
OHLCV_CACHE_PATH = "data/ohlcv_cache.db"
UNIVERSE_CACHE_PATH = "data/universe.json"
CHECKPOINT_DIR = "model/checkpoints"
LOG_DIR = "logs"
LOG_FILE = "logs/nextoc.log"
