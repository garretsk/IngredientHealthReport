import os
import io
import time
import json
import streamlit
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from openai import AzureOpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

azure_computer_vision_api_key = os.getenv("AZURE_COMPUTER_VISION_API_KEY")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

def generate_ingredient_health_report(context, health_conditions):
    client = AzureOpenAI(
        api_version="2024-10-21",
        azure_endpoint="https://excel-augmented-generation-openai.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview"
    )
    prompt = f"The following text should be from a Nutrition Facts label and contain an INGREDIENTS list: \n\n{context}\n\n\nPlease ignore all of the text other than the INGREDIENTS list and give a brief explanation of the health benefits or concerns of each individual ingredient listed for someone with the following health conditions: {health_conditions}. If the text does not contain an INGREDIENTS list, please only state that 'The image does not appear to contain an ingredient list'."
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return completion.to_json()

def ocr():
    azure_computer_vision_endpoint = "https://ocr-garret-computervision.cognitiveservices.azure.com/"
    client = ComputerVisionClient(azure_computer_vision_endpoint, CognitiveServicesCredentials(azure_computer_vision_api_key))

    image = streamlit.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image is not None:
        image = Image.open(image)
        streamlit.image(image, caption="Uploaded Image", use_container_width=True)
        streamlit.write("")
        streamlit.write("Generating report...")

        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image.format)
        image_bytes = image_bytes.getvalue()

        response = client.read_in_stream(io.BytesIO(image_bytes), raw=True)
        operation_location_remote = response.headers["Operation-Location"]
        operation_id = operation_location_remote.split("/")[-1]

        while True:
            results = client.get_read_result(operation_id)
            if results.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        if results.status == "succeeded":
            context = ""
            for text_result in results.analyze_result.read_results:
                for line in text_result.lines:
                    context += line.text + "\n"
            return context
        else:
            streamlit.write("Failed with status: {}".format(results.status))
            return None

if __name__ == "__main__":
    streamlit.title("Ingredient Health Report with Azure Computer Vision")

    inputs = []
    health_conditions = []
    inputs.append(streamlit.text_input("Enter health condition 1 (optional): "))
    inputs.append(streamlit.text_input("Enter health condition 2 (optional): "))
    inputs.append(streamlit.text_input("Enter health condition 3 (optional): "))
    inputs.append(streamlit.text_input("Enter health condition 4 (optional): "))
    for input in inputs:
        if input != "":
            health_conditions.append(input)
    
    health_conditions_text = ""
    for health_condition in health_conditions:
        if health_condition == health_conditions[-1]:
            health_conditions_text += health_condition
        else:
            health_conditions_text += health_condition + ", "
    if health_conditions_text == "":
        health_conditions_text = "None"

    context = ocr()
    if context is not None:
        summary = json.loads(generate_ingredient_health_report(context, health_conditions_text))
        streamlit.write(summary["choices"][0]["message"]["content"])