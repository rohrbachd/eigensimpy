from llama_index import SimpleDirectoryReader, ServiceContext, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os
from IPython.display import Markdown, display

os.environ["OPENAI_API_KEY"] = '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'
def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    # index = GPTSimpleVectorIndex(
    #     documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    # )

    index.save_to_disk('index.json')

    return index

def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True: 
        query = input("What do you want to ask? ")
        response = index.query(query, response_mode="compact")
        print(response.response)
        # display(Markdown(f"Response: <b>{response.response}</b>"))


# construct_index("docs")
ask_ai()