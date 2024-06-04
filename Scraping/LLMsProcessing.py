from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from langchain_pinecone import PineconeVectorStore
from pathlib import Path
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings,OpenAI
from langchain.schema import Document
from langchain_openai import ChatOpenAI 
import replicate
from langchain_experimental.chat_models import Llama2Chat
from langchain_community.llms import LlamaCpp
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.llms.huggingface import HuggingFaceLLM

os.environ['OPENAI_API_KEY'] = "sk-proj-t3quEkktCcLSjpmFpUKNT3BlbkFJSyeBPFQno4NVMSw3P1BP"
os.environ['PINECONE_API_KEY'] = '085005dc-bd04-45c5-9636-09dfc1c13a79'
os.environ['REPLICATE_API_TOKEN']='r8_ErElKYzCF5k3El1f2993fv7YRlcH9zG0f890Q'
os.environ['HUGGINGFACE_API_TOKEN']='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'

# Function to read files and create chunks
def read_files_and_create_chunks(directory, chunk_size=800, chunk_overlap=50):
    counter=0
    documents = []
    # Read the content of pages that stored in files 
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()

            doc_obj = Document(page_content=text)

            # Divied the file content into chunks
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            doc=text_splitter.split_documents([doc_obj])
            documents.extend(doc)
            
    return documents


# To create chunks
folderPath=Path("ScraptingData")
documents = read_files_and_create_chunks(folderPath)
print(len(documents))

# To create embeddings for each chunk
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print(embeddings)


def store_embedding_to_DB(index_name,embeddings,documents):
    index= PineconeVectorStore.from_documents(
            documents,
            embedding=embeddings,
            index_name=index_name
        )
    return index

index_name="data"

# # to store the embeded data # #
# store_embedding_to_DB(index_name,embeddings,documents)
# print("finished embedding")


#to retrive data
index= PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings )


template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)



#cosine similarity ret results from vector DB
def retrieve_query(query,k=2):
    matching_results= index.similarity_search(query,k=k)
    return matching_results


gpt3_llm = ChatOpenAI(  
    model_name='gpt-3.5-turbo',  
    temperature=0.5 
)  
gpt3_chain= RetrievalQA.from_chain_type(
    llm=gpt3_llm,
    chain_type="stuff",
    retriever=index.as_retriever()
)


gpt4_llm = ChatOpenAI(  
    model_name='gpt-4',  
    temperature=0.5 
) 
gpt4_chain= RetrievalQA.from_chain_type(
    llm=gpt4_llm,
    chain_type="stuff",
    retriever=index.as_retriever()
)

# # Load Llama-2 model and tokenizer
# model_name = "meta-llama/Llama-2-70b-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Define Llama-2 LLM for LangChain
# llama2_llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)

# # Create retrieval QA chain for Llama-2
# Llama2_chain = RetrievalQA.from_chain_type(
#     llm=llama2_llm,
#     chain_type="stuff",
#     retriever=index.as_retriever()
# )

# # Load Falcon-40b-instruct model and tokenizer
# falcon_model_name = "tiiuae/falcon-40b-instruct"
# falcon_tokenizer = AutoTokenizer.from_pretrained(falcon_model_name)
# falcon_model = AutoModelForCausalLM.from_pretrained(falcon_model_name)
# falcon_llm = HuggingFaceLLM(model=falcon_model, tokenizer=falcon_tokenizer)

# # Create retrieval QA chain for Falcon-40b-instruct
# Falcon_chain = RetrievalQA.from_chain_type(
#     llm=falcon_llm,
#     chain_type="stuff",
#     retriever=index.as_retriever()
# )


# Search for the answer from vector DB
def retrieve_answers(query,chainmodel):
    doc_search= retrieve_query(query)
    print(doc_search)
    response=chainmodel.invoke(query)
    return response

our_query="Can you name all the eServices?"
gpt3_answer= retrieve_answers(our_query,gpt3_chain)
gpt4_answer= retrieve_answers(our_query,gpt4_chain)

# print(Llama2_llm_answer['result'])
# print(gpt4_answer['result'])


# # Retrieve answer using Falcon-40b-instruct
# Falcon_llm_answer = retrieve_answers(our_query, Falcon_chain)
# print("Falcon-40b Answer:", Falcon_llm_answer['result'])
