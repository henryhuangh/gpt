from llama_index import GPTListIndex, GPTSimpleVectorIndex, SimpleDirectoryReader, Document, LLMPredictor, ServiceContext
from custom_LLM import CustomLLM, prompt_helper
from gptcache.embedding.huggingface import Huggingface
from llama_index import ServiceContext, LangchainEmbedding
# load in HF embedding model from langchain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from custom_Embedding import CustomEmbedding

llm_predictor = LLMPredictor(llm=CustomLLM())
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name='all-mpnet-base-v2'))


service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)
# Load documents into index
documents = SimpleDirectoryReader(input_files=[
                                  r"D:\Gigabyte\Downloads\DAD_EquityFrictions_SLIDES.pdf"]).load_data()
index = GPTSimpleVectorIndex.from_documents(
    documents, service_context=service_context)

index.save_to_disk('EquityFrictions.json')
