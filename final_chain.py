import time
from langchain_community.vectorstores import FAISS 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

start = time.time()

# LLM MODEL WITH OLLAMA 
def load_llm(model_name):
    LLM = ChatOllama(
        model = model_name,
        temperature = 0.1,
        num_predict = 256
    )
    return LLM

LLM = load_llm("vi-dominic/vistral:7b")



# PROMPT 
prompt_template = """
Bạn là một trợ lý AI hữu ích của một cửa hàng cà phê Starbucks. Dưới đây là một số thông tin từ tài liệu về Starbucks. Hãy sử dụng chúng để trả lời câu hỏi của khách hàng. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời. Hãy giữ câu trả lời ngắn gọn.
Thông tin: {context}
Câu hỏi: {question}
Trả lời:
"""

def create_prompt(prompt_template):
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    return prompt 

prompt = create_prompt(prompt_template=prompt_template)



# VECTOR STORE 
vector_db_path = r"vectorstore\faiss_db"

def load_vectorstore(model_name: str):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vector_db = FAISS.load_local(folder_path=vector_db_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    return vector_db 


vectorstore = load_vectorstore(model_name="hiieu/halong_embedding")

retriever = vectorstore.as_retriever()

cross_encoder_model = HuggingFaceCrossEncoder(model_name="namdp-ptit/ViRanker")
compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=3)
compressor_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


# RAG CHAIN 
def format_docs(documents):
    return "\n\n".join(document.page_content for document in documents)

rag_chain = (
    {"context": compressor_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | LLM
    | StrOutputParser()
)

# while True:
#     user_input = input("\nHãy nhập câu hỏi của bạn: \n")
#     if user_input.lower() == "end":
#         break   
#     for chunk in rag_chain.stream(user_input):
#         print(chunk, end="", flush=True)

user_input = input("\nNhập câu hỏi của bạn: \n")     
result = rag_chain.invoke(user_input)
print(f"\nAnswer: {result}\n")
context = compressor_retriever.invoke(user_input)
print(f"Context: {context[0]}\n")

end = time.time()
print(f"\nChạy trong {round(end-start)}s")