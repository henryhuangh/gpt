import json
from urllib import response
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GPTNeoForCausalLM, AutoTokenizer, pipeline

model = LlamaForCausalLM.from_pretrained(
    "vicuna-7B", torch_dtype=torch.float16, device_map='auto', load_in_8bit=True)
tokenizer = LlamaTokenizer.from_pretrained("vicuna-7B")
generator = pipeline("text-generation", model=model,
               tokenizer=tokenizer)



app = Flask(__name__)
cors = CORS(app)


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
def live():
    return "live"


if __name__ in "__main__":
    app.run()