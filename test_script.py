import os
from backend.image_processor import image_processor
from backend.llava_med import model as med_model

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

message = "What abnormalities you can see in this image?"

med_responces = []
input_ids, attention_mask = med_model.create_conversations(message)

filename = "/kaggle/input/demo-hne/demo_HNE.png"
filepath = os.path.join(UPLOAD_FOLDER, filename)

image_processor.analyze_hne_image(filepath)     # patches are saved in temp_uploads folder

for image_name in os.listdir(UPLOAD_FOLDER):
    image_path = os.path.join(UPLOAD_FOLDER, image_name)
    image_tensor = med_model.get_image_tensors(image_path)
    answer = med_model.generate_response(input_ids, attention_mask, image_tensor)
    med_responces.append({image_path : answer})

print(med_responces)


import os
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from uuid import uuid4
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def get_response(image_analysis_data):
    # Combine image descriptions into a formatted string
    formatted_image_info = "\n".join(
        [f"{filename}:\n{description}" for item in image_analysis_data for filename, description in item.items()]
    )

    # Initialize the LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_retries=2,
        api_key="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"
    )

    # Define the prompt
    prompt_template = PromptTemplate(
        input_variables=["embeddings"],
        template="""
You are a medical pathology assistant AI.

Below are descriptions of histopathology image patches. Each description corresponds to an image filename. Analyze all the provided information and produce a **concise overall summary** of the pathological findings across the images.

Descriptions:
{image_info}

Please provide a comprehensive summary.
"""
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the chain
    response = chain.run({
        "image_info": formatted_image_info
    })

    return response

# ChatGroq.model_rebuild()

responce = get_response(med_responces)
print(responce)