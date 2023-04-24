from transformers import GPTJForCausalLM, LlamaForCausalLM
import torch
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# save model
embed_model.save('all-mpnet-base-v2')

llm_model = LlamaForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-delta-v1.1", torch_dtype=torch.float16, low_cpu_mem_usage=True)
# save model
llm_model.save_pretrained("vicuna-7b-delta-v1.1")
