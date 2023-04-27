from llama_index import GPTListIndex, LLMPredictor, ServiceContext, GPTSimpleVectorIndex
from custom_LLM import CustomLLM, prompt_helper
from llama_index import ServiceContext, LangchainEmbedding
# load in HF embedding model from langchain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.llamacpp import LlamaCppEmbeddings

llm_predictor = LLMPredictor(llm=CustomLLM())
embed_model = LangchainEmbedding(
    LangchainEmbedding(model_name='llama-7B'))


service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)


index = GPTSimpleVectorIndex.load_from_disk(
    'gpt-pdf.json', service_context=service_context)

response = index.query("What is a competitve equilibrium?", mode="default",
                       verbose=True)
print(response)
