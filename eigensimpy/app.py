from llama_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext

from langchain import OpenAI
import openai
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    #openai.
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    documents = SimpleDirectoryReader(directory_path).load_data()
    
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    
    #index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)