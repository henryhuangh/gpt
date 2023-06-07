from custom_LLM import CustomLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


llm = CustomLLM(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

conversation.predict(input="Hello there")
