from langchain.llms.base import LLM
from transformers import pipeline
from typing import Optional, List, Mapping, Any
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, GPTNeoForCausalLM, GPT2LMHeadModel, GPT2TokenizerFast
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from threading import Thread
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, pipeline, logging, AutoTokenizer, TextGenerationPipeline, TextStreamer, TextIteratorStreamer

# model = LlamaForCausalLM.from_pretrained(
#     "vicuna-13B", torch_dtype=torch.float16, device_map='auto', load_in_8bit=True)
# tokenizer = LlamaTokenizer.from_pretrained("vicuna-13B")
# generator = pipeline("text-generation", model=model,
#                      tokenizer=tokenizer)

quantized_model_dir = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit.act.order"
# quantized_model_dir = "TheBloke/guanaco-7B-GPTQ"
# model_basename = "Guanaco-7B-GPTQ-4bit-128g.no-act-order"


quantize_config = BaseQuantizeConfig(
    bits=4,
    # group_size=128,
    desc_act=False
)

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir,
                                           use_safetensors=True,
                                           model_basename=model_basename,
                                           device="cuda:0",
                                           use_triton=False,
                                           quantize_config=quantize_config).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=True)
generator = TextGenerationPipeline(
    model=model, tokenizer=tokenizer)


class CustomLLM(LLM):
    model_name = 'Wizard-Vicuna-30B'
    streaming: bool = False

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None,) -> str:
        prompt_length = len(prompt)
        inputs = tokenizer([prompt], return_tensors="pt").input_ids.to('cuda')
        generator = TextGenerationPipeline(
            model=model, tokenizer=tokenizer)
        if self.streaming:
            stream_resp = TextIteratorStreamer(tokenizer, skip_prompt=True)
            generation_kwargs = dict(
                inputs=inputs, streamer=stream_resp, max_new_tokens=1024, no_repeat_ngram_size=2,
                early_stopping=True)

            thread = Thread(target=model.generate,
                            kwargs=generation_kwargs, daemon=True)
            # model.generate(streamer=streamer, max_new_tokens=256)
            thread.start()
            completion = ''
            for data in stream_resp:
                completion += data
                if run_manager:
                    run_manager.on_llm_new_token(data)
            return completion
        generator = TextGenerationPipeline(
            model=model, tokenizer=tokenizer)
        response = generator(prompt)
        text = response[0]["generated_text"]
        return text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"
