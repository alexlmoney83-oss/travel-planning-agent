"""
全局配置文件
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 加载环境变量（明确指定.env文件路径）
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# DeepSeek API配置（用于R1推理模型）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# DashScope API配置（用于Qwen模型和Embedding）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN3_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# LangSmith配置（可选，仅用于调试）
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

# 模型配置
QWEN3_MODEL = "qwen-plus"  # 使用DashScope API
QWEN3_TEMPERATURE = 0.7

R1_MODEL = "deepseek-reasoner"
R1_TEMPERATURE = 0.1

# Embedding模型
EMBEDDING_MODEL = "text-embedding-v4"

# RAG配置
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "travel_vectordb"
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
RAG_SEARCH_K = 3
RAG_BATCH_SIZE = 10  # ChromaDB批量载入大小，如遇到API限制可调小

# MCP配置
_mcp_path_env = os.getenv("MCP_CONFIG_PATH", "travel_agent/config/servers_config.json")
# 如果是相对路径，则相对于PROJECT_ROOT解析
if not Path(_mcp_path_env).is_absolute():
    MCP_CONFIG_PATH = str(PROJECT_ROOT / _mcp_path_env)
else:
    MCP_CONFIG_PATH = _mcp_path_env
