from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, Document, LLMPredictor, LangchainEmbedding, ServiceContext, QuestionAnswerPrompt, StorageContext, load_index_from_storage

from custom_LLM import CustomLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

llm = LLMPredictor(CustomLLM())
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))


query_str = "What did the author do growing up?"
QA_PROMPT_TMPL = (
    "We have provided context information below. \n"
    "---------------------\n"
    "Context: {context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\nAnswer:"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

service_context = ServiceContext.from_defaults(llm_predictor=llm, embed_model=embed, chunk_size_limit=900)
storage_context = StorageContext.from_defaults(persist_dir='ac_lg_design')
# index = GPTVectorStoreIndex.from_documents(service_context=service_context, storage_context=storage_context)
index=load_index_from_storage(storage_context=storage_context, service_context=service_context)
index._include_extra_info = True
# query_engine = index.as_query_engine()
query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)

response = query_engine.query("Who is the author?")
print(response)

response = query_engine.query("What is a tire?")
print(response)

response = query_engine.query("What is the recommended growth capability for sizing a tire?")
print(response)

response = query_engine.query("What is a CBR?")
print(response)

response = query_engine.query("When was CBR developed?")
print(response)