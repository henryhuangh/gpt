import json
from urllib import response
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from flask_caching import Cache
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GPTNeoForCausalLM, AutoTokenizer, pipeline, LlamaTokenizer
import uuid
from custom_LLM import CustomLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, Document, LLMPredictor, LangchainEmbedding, ServiceContext, QuestionAnswerPrompt, StorageContext, load_index_from_storage
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

llm = LLMPredictor(CustomLLM())
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))


QA_PROMPT_TMPL = (
    "We have provided context information below. \n"
    "---------------------\n"
    "Context: {context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\nAnswer:"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

service_context = ServiceContext.from_defaults(llm_predictor=llm, embed_model=embed, chunk_size_limit=900)
storage_context = StorageContext.from_defaults(persist_dir='ac_lg_design')
# index = GPTVectorStoreIndex.from_documents(service_context=service_context, storage_context=storage_context)
index=load_index_from_storage(storage_context=storage_context, service_context=service_context)
index._include_extra_info = True
# query_engine = index.as_query_engine()
query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)


config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
app = Flask(__name__, static_folder='build', static_url_path='')
cors = CORS(app)

app.config.from_mapping(config)
cache = Cache(app)


@app.route('/', methods=['GET'])
@cross_origin()
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    data = request.get_json()

    answer = query_engine.query(data["response"])
    return jsonify({"response": answer.response, 'chat_id': '0'})




if __name__ in "__main__":
    app.run(host='0.0.0.0', port=80)