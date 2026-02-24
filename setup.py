from setuptools import find_packages, setup


setup(
    name="medical-chatbot-with-llm",
    version="0.0.1",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
)
install_requires=[
    "flask==3.1.1",
    "gunicorn==21.2.0",
    "langchain==0.3.26",
    "langchain-community==0.3.26",
    "langchain-openai==0.3.24",
    "langchain-pinecone==0.1.3",
    "langchain-nvidia-ai-endpoints==0.3.2",
    "sentence-transformers==4.1.0",
    "pypdf==5.6.1",
    "python-dotenv==1.1.0"
]