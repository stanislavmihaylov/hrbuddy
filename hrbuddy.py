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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # with GPU
    # torch.cuda.empty_cache()
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16,quantization_config=bnb_config)

    model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16)

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2048,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return model, text_generation_pipeline

def init_llm_chain(text_generation_pipeline):
    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
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
    return LLMChain(llm=mistral_llm, prompt=prompt)


try: 
    docs = get_confluence_view_content()
    client, db = init_db(docs)
    model, text_generation_pipeline = init_model()
    llm_chain = init_llm_chain(text_generation_pipeline)
    retriever = db.as_retriever()

    rag_chain = ( 
    {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

    while True:
        user_question = input("Ask me something: (type 'quit' to end conversation)")
        if user_question == "quit":
            break
        else:
            answer = rag_chain.invoke(user_question)['text']
            print(answer)

except OSError as err:
    print("OS error:", err)
finally: 
    client.close()
    # torch.cuda.empty_cache()
    del model
    del pipeline
    gc.collect()