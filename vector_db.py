from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Khai báo biến cho Vector DB
file_path = r"data"
vector_db_path = r"vectorstore/faiss_db"
text_loader_kwargs={'autodetect_encoding': True}

def create_vector_db_from_files():
    # Sử dụng loader để quét toàn bộ file trong folder "data"
    loader = DirectoryLoader(path=file_path, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    documents = loader.load()
    
    # Chia các đoạn văn bản trong các file thành chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = text_splitter.split_documents(documents=documents)
    
    # Embedding các chunks và nạp vào Vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="hiieu/halong_embedding")
    #embedding_model = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    vector_db.save_local(vector_db_path)
    
    return vector_db

# Gọi hàm để tạo Vector DB
create_vector_db_from_files()