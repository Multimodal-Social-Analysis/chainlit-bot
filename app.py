from dotenv import load_dotenv
from openai import AsyncOpenAI
import chainlit as cl
import os
from langchain.chains.llm import LLMChain
#from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain.memory.buffer import ConversationBufferMemory
from prompts import general_prompt, test_prompt
#from textblob import TextBlob
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# env
load_dotenv()
# HUGGINGFACEHUB_API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = AsyncOpenAI()
cl.instrument_openai

factors = [
  "Commerce",
  "Culture",
  "Drugs",
  "Education",
  "Gastronomy",
  "Infrastructure",
  "Nature",
  "Poverty",
  "Pollution",
  "Religion",
  "Security",
  "Sexuality",
  "Socialization"
]

# continuously on a loop
@cl.on_message
async def main(message: cl.Message):
  if ("analyze text") in message.content.lower():
    # get text
    file = None
    while file == None:
      file = await cl.AskFileMessage(content="Please upload a text file to analyze", accept={"text/plain": [".txt", ".md"]}).send()
    file = file[0]

    with open(file.path, 'r', encoding="utf-8") as f:
      text = f.read()

    await cl.Message(content=text).send()

    input = f"{text}\nFactors: {factors}\n"

    response = await client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": "Choose one of the following factors that best applies to this text (only respond with the answer).\nDon't justify your answers. Don't give information not mentioned in the context information."
        },
        {
          "role": "user",
          "content": input,
        }
      ],
    )

    await cl.Message(content=response.choices[0].message.content).send() # response.choices[0].message.content
  else:
    input = f"{message.content}\nFactors: {factors}"

    response = await client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": "Choose one of the following factors that best applies to this text (only respond with the answer).\nDon't justify your answers. Don't give information not mentioned in the context information."
        },
        {
          "role": "user",
          "content": input,
        }
      ],
    )

    await cl.Message(content=response.choices[0].message.content).send()