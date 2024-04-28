from langchain.prompts import PromptTemplate

test = """
Text:
{query}
Factors: [Economic, Health, Judicial, Religion]
Which one following factors does this text best represents?

Answer:
"""
test_prompt = PromptTemplate(template=test, input_variables=['query'])

general_template = """
You are a multimodal social analysis AI assistant that analyzes a text based on the query received as input.
Your expertise is in analyzing each sentences based on factors and aspect, and attaching the factors or aspects to the sentences that are related.

If a question is not about analyzing factors or aspect, respond with, "I specialize only in multimodal social analysis related queries."

Question: {query}
Answer:
"""
general_prompt = PromptTemplate(template=general_template, input_variables=['chat_history', 'query'])


# ChatPromptTemplate.from_messages([
#   [
#     "system",
#     "You are a helpful assistant that gets the most likely factors from the text.",
#   ],
#   ["user", "{text}"],
# ])