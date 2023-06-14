import json
from urllib import response
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS, cross_origin
from flask_caching import Cache
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GPTNeoForCausalLM, AutoTokenizer, pipeline, LlamaTokenizer
import uuid
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.callbacks.base import BaseCallbackHandler
from threading import Thread
from custom_LLM import CustomLLM, model, tokenizer, generator
from transformers import TextIteratorStreamer


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
    if request.headers.get('accept') == 'text/event-stream':
        def events():
            inputs = tokenizer(
                [data['query']], return_tensors='pt').input_ids.to('cuda')
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
            generation_kwargs = dict(
                inputs=inputs, streamer=streamer, max_new_tokens=1024, no_repeat_ngram_size=2, early_stopping=True)
            thread = Thread(target=model.generate,
                            kwargs=generation_kwargs, daemon=True)
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield f"{new_text.replace(tokenizer.eos_token, ' ')}\n\n"
        return Response(events(), content_type='text/event-stream')
    else:
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
        conversation_mem = cache.get(data["chat_id"])
        chat_id = data["chat_id"]
    else:
        conversation_mem = None
        chat_id = str(uuid.uuid4())

    if conversation_mem is None:
        conversation_mem = ConversationBufferMemory()

    if request.headers.get('accept') == 'text/event-stream':
        from queue import Queue
        q = Queue(1)

        class CustomHandler(BaseCallbackHandler):
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                q.put(token)

        llm = CustomLLM(streaming=True, callbacks=[CustomHandler()])
        conversation = ConversationChain(
            llm=llm,
            memory=conversation_mem
        )
        done = object()

        def resolve_convo(input):
            conversation.predict(input=input)
            cache.set(chat_id, conversation.memory)
            q.put(done)

        input = data["response"]
        thread = Thread(target=resolve_convo, args=(input, ), daemon=True)
        thread.start()

        def stream():
            while True:
                new_text = q.get(timeout=8)
                if new_text is done or 'human:' in new_text:
                    break
                yield new_text.replace(tokenizer.eos_token, '')
        return Response(stream(), content_type='text/event-stream')
    else:
        llm = CustomLLM()
        conversation = ConversationChain(
            llm=llm,
            memory=conversation_mem
        )

        response = conversation.predict(input=data["response"])
        cache.set(chat_id, conversation.memory)
        return jsonify({"response": response, "chat_id": chat_id})


@app.route('/get_mem', methods=['POST'])
@cross_origin()
def get_mem():
    data = request.get_json()
    conversation_mem = cache.get(data["chat_id"])
    return jsonify(conversation_mem.json())


if __name__ in "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)
