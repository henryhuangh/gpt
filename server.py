import json
from urllib import response
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from flask_caching import Cache
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GPTNeoForCausalLM, AutoTokenizer, pipeline
import uuid
from custom_LLM import CustomLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory


model = LlamaForCausalLM.from_pretrained(
    "vicuna-7B", torch_dtype=torch.float16, device_map='auto', load_in_8bit=True)
tokenizer = LlamaTokenizer.from_pretrained("vicuna-7B")
generator = pipeline("text-generation", model=model,
               tokenizer=tokenizer)


config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
app = Flask(__name__, static_folder='build', static_url_path='')
cors = CORS(app)

app.config.from_mapping(config)
cache = Cache(app)

@app.route('/text_generation', methods=['POST'])
@cross_origin()
def text_generation():
    data = request.get_json()
    response = jsonify(
        generator(data["query"], 
        do_sample=True, 
        temperature=data.get("temperature", 0.9), 
        max_new_tokens=data.get("max_new_tokens", 256)))
    return response

@app.route('/', methods=['GET'])
@cross_origin()
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    data = request.get_json()
    if "chat_id" in data:
        conversation = cache.get(data["chat_id"])
        chat_id = data["chat_id"]
    else:
        conversation = None
        chat_id = str(uuid.uuid4())
    
    if conversation is None:
        llm = CustomLLM()
        memory = ConversationSummaryMemory(llm=llm)
        conversation = ConversationChain(
            llm=llm, 
            memory=memory
        )

    response = conversation.predict(input=data["response"])

    cache.set(chat_id, conversation)
    return jsonify({"response": response, "chat_id": chat_id})

@app.route('/get_mem', methods=['POST'])
@cross_origin()
def get_mem():
    data = request.get_json()
    conversation = cache.get(data["chat_id"])
    return jsonify(conversation.memory.json())


if __name__ in "__main__":
    app.run(host='0.0.0.0', port=80)