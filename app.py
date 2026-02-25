from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
NVIDI_AI_API_KEY=os.environ.get('NVIDIA_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["NVIDIA_API_KEY"] = NVIDI_AI_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

load_dotenv()
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatNVIDIA(model="meta/llama3-8b-instruct")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('index.html')





@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    msg = data["msg"]

    print("USER:", msg)

    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]

    print("BOT:", answer)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
 