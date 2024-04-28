import os
from dotenv import load_dotenv
import chainlit as cl
from langchain.chains.llm import LLMChain
#from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain.memory.buffer import ConversationBufferMemory
from prompts import general_prompt, factors_prompt, aspects_prompt, test_prompt
#from textblob import TextBlob
from transformers import AutoModelForCausalLM, AutoTokenizer

#! model
#! output
#! publish

# env
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']

# transformer/pre-trained model
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Default prompt is both factors and aspects
prompt = general_prompt

# Model type
model_id = "gpt2-medium"
conv_model = HuggingFaceHub(
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    repo_id=model_id,
    model_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 50
    }
)
conversation_memory = ConversationBufferMemory(memory_key="chat_history", max_len=50, return_messages=True)

# Chat Bot
conv_chain = LLMChain(
  llm=conv_model,
  prompt=prompt,
  memory=conversation_memory,
  verbose=True
)

aspects = ""

factors = ""

# what -> factors/aspects -> example -> text -> input/user msg -> output

# continuously on a loop
@cl.on_message
async def main(message: cl.Message):
  input = message.content.lower()

  # factors
  if ("upload factors" or "upload factor") in input:
    get_factors_file = await cl.AskFileMessage(content="Please upload a new list of factors to analyze", accept={"text/plain": [".txt", ".md"]}).send()
    factors_file = get_factors_file[0]

    with open(factors_file.path, 'r', encoding="utf-8") as f:
      factors = f.read().strip().split(',')
    #assert isinstance(factors, list)

  # aspects
  elif ("upload aspects" or "upload aspect") in input:
    get_aspects_file = await cl.AskFileMessage(content="Please upload a new list of aspects to analyze", accept={"text/plain": [".txt", ".md"]}).send()
    aspects_file = get_aspects_file[0]

    with open(aspects_file.path, 'r', encoding="utf-8") as f:
      aspects = f.read().strip().split(',')
    #assert isinstance(factors, list)

  # analyze text
  elif ("analyze text") in input:
    # get file
    file = None
    while file == None:
      file = await cl.AskFileMessage(content="Please upload a text file to analyze", accept={"text/plain": [".txt", ".md"]}).send()
    text_file = file[0]

    # get text to analyze
    with open(text_file.path, 'r', encoding="utf-8") as f:
      text = f.read()
    #msg = cl.Message(content=text)
    #await msg.send()

    # specified
    if ("based on") in input:
      if ("factors" or "factor") in input:
        decision = await cl.AskActionMessage(
          content="Use existing factors or upload a new one?",
          actions=[
            cl.Action(name="existing", value="existing", label="existing"),
            cl.Action(name="new", value="new", label="new")
          ]
        ).send()

        if decision and decision.get("value") == "new":
          # get new factors
          new_factors_file = await cl.AskFileMessage(content="Please upload a new factors file", accept={"text/plain": [".txt", ".md"]}).send()
          new_factors = new_factors_file[0]

          with open(new_factors.path, 'r', encoding="utf-8") as f:
            factors = f.read()
          factors = "Factors: " + "[" + factors + "]"
          #assert isinstance(factors, str)

          #! check if its factors
          #factors_msg = cl.Message(content=factors)
          #await factors_msg.send()

        # bot
        new_input = """
          You are a multimodal social analysis AI assistant that analyzes a given text received as input based on a given factors or aspect.
          Your expertise is in analyzing each sentences and words based on factors and aspect, and attaching the factors or aspects to the words that are related in this format: (words)[factor] or (words)[aspect].

          Text:
          {file}

          Factors:
          {factors}

          Input:
          Analyze text based on factors

          Output:
        """

        input_ids = tokenizer.encode(new_input, return_tensors="pt")
        output = model.generate(input_ids, max_length=2480, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        mes = cl.Message(content=generated_text)
        await mes.send()
      elif ("aspects" or "aspect") in input:
        decision = await cl.AskActionMessage(
          content="Use existing aspects or upload a new one?",
          actions=[
            cl.Action(name="existing", value="existing", label="existing"),
            cl.Action(name="new", value="new", label="new")
          ]
        ).send()

        if decision and decision.get("value") == "new":
          # get new aspects
          new_aspects_file = await cl.AskFileMessage(content="Please upload a new aspects file", accept={"text/plain": [".txt", ".md"]}).send()
          new_aspects = new_aspects_file[0]

          with open(new_aspects.path, 'r', encoding="utf-8") as f:
            aspects = f.read()
          aspects = "Aspects: " + "[" + aspects + "]"
          #assert isinstance(aspects, str)

          #! check if its aspects
          #aspects_msg = cl.Message(content=aspects)
          #await aspects_msg.send()

          # bot

        # existing
        else:
          # bot
          None
      # not factors/aspects
      else:
        # stop
        None
    # not specified
    else:
      # choose based on factors or aspects
      res = await cl.AskActionMessage(
        content="Analyze Factors or Aspects?",
        actions=[
          cl.Action(name="factors", value="factors", label="factors"),
          cl.Action(name="aspects", value="aspects", label="aspects")
        ]
      ).send()

      if res and res.get("value") == "factors":
        decision = await cl.AskActionMessage(
          content="Use existing factors or upload a new one?",
          actions=[
            cl.Action(name="existing", value="existing", label="existing"),
            cl.Action(name="new", value="new", label="new")
          ]
        ).send()

        if decision and decision.get("value") == "new":
          # get new aspects
          new_factors_file = await cl.AskFileMessage(content="Please upload a new factors file", accept={"text/plain": [".txt", ".md"]}).send()
          new_factors = new_factors_file[0]

          with open(new_factors.path, 'r', encoding="utf-8") as f:
            factors = f.read()
          factors = "Factors: " + "[" + factors + "]"
          #assert isinstance(aspects, str)

          #! check if its aspects
          #aspects_msg = cl.Message(content=aspects)
          #await aspects_msg.send()

          # bot

        # existing
        else:
          # bot
          None
      # aspects
      else:
        decision = await cl.AskActionMessage(
          content="Use existing aspects or upload a new one?",
          actions=[
            cl.Action(name="existing", value="existing", label="existing"),
            cl.Action(name="new", value="new", label="new")
          ]
        ).send()

        if decision and decision.get("value") == "new":
          # get new aspects
          new_aspects_file = await cl.AskFileMessage(content="Please upload a new aspects file", accept={"text/plain": [".txt", ".md"]}).send()
          new_aspects = new_aspects_file[0]

          with open(new_aspects.path, 'r', encoding="utf-8") as f:
            aspects = f.read()
          aspects = "Aspects: " + "[" + aspects + "]"
          #assert isinstance(aspects, str)

          #! check if its aspects
          #aspects_msg = cl.Message(content=aspects)
          #await aspects_msg.send()

          # bot

        # existing
        else:
          # bot
          None
  # reject/unable
  else:
    # llm_chain = cl.user_session.get("llm_chain")

    # #! no need prompt: add -> input
    # res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]) # use the template to ask question
    # query = res["query"] # user input
    # text = res["text"] # bot output

    # msg = cl.Message(content=text)
    # await msg.send()

    new_input = """
      You are a multimodal social analysis AI assistant that analyzes a given text received as input based on a given factors or aspect.
      Your expertise is in analyzing each sentences and words based on factors and aspect, and attaching the factors or aspects to the words that are related in this format: (words)[factor] or (words)[aspect].

      If a question is not about analyzing factors or aspect, I respond with, "I specialize only in multimodal social analysis related queries."
    """
    input_ids = tokenizer.encode(new_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    mes = cl.Message(content=generated_text)
    await mes.send()

@cl.on_chat_start
async def start():
  cl.user_session.set("llm_chain", conv_chain)

  # ask factors
  get_factors_file = await cl.AskFileMessage(content="Please upload a list of factors to analyze", accept={"text/plain": [".txt", ".md"]}).send()
  factors_file = get_factors_file[0]

  with open(factors_file.path, 'r', encoding="utf-8") as f:
    factors = f.read()

  factors = "Factors: " + "[" + factors + "]"
  #assert isinstance(factors, str)

  #! check if its factors
  factors_msg = cl.Message(content=factors)
  await factors_msg.send()

  # ask aspects
  get_aspects_file = await cl.AskFileMessage(content="Please upload a list of aspects to analyze", accept={"text/plain": [".txt", ".md"]}).send()
  aspects_file = get_aspects_file[0]

  with open(aspects_file.path, 'r', encoding="utf-8") as f:
    aspects = f.read()

  aspects = "Aspects: " + "[" + aspects + "]"
  #assert isinstance(factors, list)

  #! check if its aspects
  aspects_msg = cl.Message(content=aspects)
  await aspects_msg.send()