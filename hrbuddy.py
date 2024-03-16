import gc
import os
import torch
import requests
from requests.auth import HTTPBasicAuth
import json
import html2text
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
import weaviate
from weaviate.auth import AuthApiKey
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain

_ = load_dotenv(find_dotenv())

def get_confluence_view_content():
    url = f"{os.environ.get("CONFLUENCE_BASE_URL")}/wiki/api/v2/pages/4626481153?body-format=view"
    auth = HTTPBasicAuth("stanislav.mihaylov@mentormate.com", os.environ.get("CONFLUENCE_API_KEY"))
    headers = {
    "Accept": "application/json"
    }

    response = requests.request(
    "GET",
    url,
    headers=headers,
    auth=auth
    )

    return json.loads(response.text)['body']['view']['value']

def split_docs(docs): 
    extracted_text = html2text.html2text(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    return text_splitter.create_documents([extracted_text])

def init_db(docs):
    client = weaviate.connect_to_wcs(
        cluster_url=os.environ.get("WEAVIATE_BASE_URL"),
        auth_credentials=AuthApiKey(os.environ.get("WEAVIATE_API_KEY")),
        skip_init_checks=True
    )
    db = WeaviateVectorStore.from_documents(split_docs(docs), HuggingFaceEmbeddings( model_name="sentence-transformers/all-mpnet-base-v2"), client=client)

    return client, db


def init_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    model_config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=True,
            do_sample=True,
            max_new_tokens=1000,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
    )
    return model, text_generation_pipeline

try: 
    docs = get_confluence_view_content()
    client, db = init_db(docs)
    model, text_generation_pipeline = init_model()
    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    retriever = db.as_retriever()

    prompt_template = """
    ### [INST] Instruction: Answer the question based on the provided context. Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain 
    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    rag_chain = ( 
    {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

    while True:
        user_question = input("Ask me something:")
        if user_question == "quit":
            break
        else:
            answer = rag_chain.invoke(user_question)['text']
            print(answer)

except OSError as err:
    print("OS error:", err)
finally: 
    client.close()
    del model
    del pipeline
    gc.collect()