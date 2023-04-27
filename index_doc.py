from langchain.embeddings.self_hosted_hugging_face import SelfHostedHuggingFaceEmbeddings
from llama_index import GPTListIndex, GPTSimpleVectorIndex, SimpleDirectoryReader, Document, LLMPredictor, ServiceContext
from custom_LLM import CustomLLM, prompt_helper
from gptcache.embedding.huggingface import Huggingface
from llama_index import ServiceContext, LangchainEmbedding
# load in HF embedding model from langchain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

llm_predictor = LLMPredictor(llm=CustomLLM())
embed_model = LangchainEmbedding(
    OpenAIEmbeddings())


service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)
# Load documents into index
documents = SimpleDirectoryReader(input_files=[
                                  r"D:\Gigabyte\Documents\coding\gpt\examples\MacDev_2023_04 (3) (1).pdf"]).load_data()
index = GPTSimpleVectorIndex.from_documents(
    documents, service_context=service_context)

index.save_to_disk('gpt-pdf.json')
