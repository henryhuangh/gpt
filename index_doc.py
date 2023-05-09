from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, Document, LLMPredictor, LangchainEmbedding, ServiceContext
from custom_LLM import CustomLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

llm = LLMPredictor(CustomLLM())
model_name = "sentence-transformers/all-MiniLM-L6-v2"

embed = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))   
service_context = ServiceContext.from_defaults(llm_predictor=llm, embed_model=embed, chunk_size_limit=900)
documents = SimpleDirectoryReader(r"C:\Users\path\Documents\PDF-partial").load_data()
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
index._include_extra_info = True
index.storage_context.persist(persist_dir="ac_lg_design")

