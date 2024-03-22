import gc
import os
import requests
from requests.auth import HTTPBasicAuth
import json
import html2text
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
import weaviate
from weaviate.auth import AuthApiKey
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

_ = load_dotenv(find_dotenv(), override=True)

def get_confluence_view_content():
    print('start get documents')
    url = f"{os.environ.get("CONFLUENCE_BASE_URL")}/wiki/api/v2/pages/4626481153?body-format=view"
    auth = HTTPBasicAuth(os.environ.get("CONFLUENCE_EMAIL"), os.environ.get("CONFLUENCE_API_KEY"))
    headers = {
        "Accept": "application/json"
    }

    response = requests.request(
        "GET",
        url,
        headers=headers,
        auth=auth
    )

    print('end get documents')
    return json.loads(response.text)['body']['view']['value']

def split_docs(docs): 
    print('start split documents')
    extracted_text = html2text.html2text(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )

    print('end split documents')
    return text_splitter.create_documents([extracted_text])

def init_db(docs):
    print('start init db')
    client = weaviate.connect_to_wcs(
        cluster_url=os.environ.get("WEAVIATE_BASE_URL"),
        auth_credentials=AuthApiKey(os.environ.get("WEAVIATE_API_KEY")),
        skip_init_checks=True
    )
    db = WeaviateVectorStore.from_documents(split_docs(docs), HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'), client=client)

    print('end init db')
    return client, db

def get_llm_model():
    (repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                                "mistral-7b-instruct-v0.2.Q5_K_M.gguf")

    filepath = try_to_load_from_cache(repo_id=repo_id, filename=model_file_name, cache_dir=os.environ.get("HF_HOME"))
    
    if not isinstance(filepath, str):
        filepath = hf_hub_download(repo_id=repo_id,
                                 filename=model_file_name,
                                 repo_type="model")
    
    return filepath

def init_llama_chain(retriever):
    print('start init llm')
    llm = LlamaCpp(
        model_path=get_llm_model(),
        temperature=0,
        max_tokens=512,
        top_p=1,
        # n_gpu_layers=1,
        n_batch=512,
        n_ctx=4096,
        stop=["[INST]"],
        verbose=False,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    prompt_template = """
    ### [INST] Instruction: You are a helpful, AI powerded human resources assistant that gives information based only on the provided context. Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    print('end init llm')
    return ( 
        {
            "context": retriever, 
            "question": RunnablePassthrough()
        }
        | LLMChain(llm=llm, prompt=prompt)
    )

try: 
    docs = get_confluence_view_content()
    client, db = init_db(docs)
    retriever = db.as_retriever()
    llm_chain = init_llama_chain(retriever)

    while True:
        user_question = input("\nAsk me something: (type 'quit' to end conversation)")
        if user_question == "quit":
            break
        else:
            for chunk in llm_chain.stream(user_question):
                if(isinstance(chunk, str)):
                    print(chunk, end="|", flush=True)

except OSError as err:
    print("OS error:", err)
finally: 
    client.close()
    gc.collect()