from langchain_community.vectorstores import FAISS 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chainlit as cl



@cl.cache
# LOAD LLM
def load_llm(model_name: str):
    LLM = ChatOllama(
        model = model_name,
        temperature = 0.1,
        num_predict = 256,
        streaming=True
    )
    return LLM

LLM = load_llm("vi-dominic/vistral:7b")



# PROMPTS
prompt_template = """
Bạn là một trợ lý AI hữu ích của một cửa hàng cà phê Starbucks. Dưới đây là một số thông tin từ tài liệu về Starbucks. Hãy sử dụng chúng để trả lời câu hỏi của người dùng. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời. Hãy giữ câu trả lời ngắn gọn và không tự đặt thêm câu hỏi.
Thông tin: {context}
Câu hỏi: {question}
Trả lời:
"""

def create_prompt(prompt_template):
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    return prompt 



# VECTOR DATABASE
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




@cl.on_chat_start
async def on_chat_start():
    prompt = create_prompt(prompt_template=prompt_template)
    
    rag_chain = (
        {"context": compressor_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | LLM
        | StrOutputParser()
    )
    
    cl.user_session.set("rag_chain", rag_chain)
    
    await cl.Message(
        content="Xin chào, tôi là trợ lý ảo Starbucks, tôi có thể giúp gì cho bạn ?"
    ).send()



@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("rag_chain")   
    
    msg = cl.Message(content="")
    async for chunk in runnable.astream(message.content):
        await msg.stream_token(chunk)
    await msg.send()
    
# @cl.on_message
# async def on_message(message: str):
#     rag_chain = cl.user_session.get("rag_chain")

#     res = rag_chain.invoke(message.content)
#     await cl.Message(content=res).send()