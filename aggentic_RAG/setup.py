from setuptools import setup, find_packages

setup(
    name="travel_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.3.0",
        "langgraph>=0.2.0",
        "langchain-openai>=0.2.0",
        "langchain-community>=0.3.0",
        "langchain-chroma>=0.1.0",
        "dashscope>=1.20.0",
        "chromadb>=0.5.0",
        "python-dotenv>=1.0.0",
        "openai-agents>=0.1.0",
        "mcp>=0.1.0",
    ],
    python_requires=">=3.11",
)
