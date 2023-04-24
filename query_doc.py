from llama_index import GPTListIndex, LLMPredictor, ServiceContext, GPTSimpleVectorIndex
from custom_LLM import CustomLLM, prompt_helper
from llama_index import ServiceContext, LangchainEmbedding
# load in HF embedding model from langchain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

llm_predictor = LLMPredictor(llm=CustomLLM())
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name='all-mpnet-base-v2'))


service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)


index = GPTSimpleVectorIndex.load_from_disk(
    'EquityFrictions.json', service_context=service_context)

response = index.query("Define a competitive equilibrium?", mode="default",
                       verbose=True)
print(response)
