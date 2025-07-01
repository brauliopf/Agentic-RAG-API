"""
Centralized prompt templates for the RAG application.
"""

from langchain_core.prompts import PromptTemplate

# RAG Question-Answering Prompt
RAG_QA_TEMPLATE = """
You are an assistant for question-answering tasks. You reply strictly using the context provided. If the context provided does not contain the answer, just say that you don't know. Use the following pieces of retrieved context to answer the question. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
RAG_QA_PROMPT_TEMPLATE = PromptTemplate.from_template(RAG_QA_TEMPLATE)

# Document Grading Prompt
GRADE_DOCUMENTS_TEMPLATE = """
You are a grader assessing relevance of a retrieved document to a user question. 
Here is the retrieved document: 

{context} 

Here is the user question: {question} 

If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""
GRADE_DOCUMENTS_PROMPT_TEMPLATE = PromptTemplate.from_template(GRADE_DOCUMENTS_TEMPLATE)

# Question Rewriting Prompt
REWRITE_QUESTION_TEMPLATE = """
You are a helpful assistant that helps answer questions. You rewrite the question received to try to extract additional information from your database, to inform yourself and answer the question in a more complete way. This is the question received:
------- 
{question}
-------
Edit, but do not alter the meaning or intention of a question. Respond to this message in a structured format, with an object with a key "question" and the value set to the edited question. Do not include any other text or formatting.
For example: Ex: {{"question": "IMPROVED QUESTION?"}}
"""
REWRITE_QUESTION_PROMPT_TEMPLATE = PromptTemplate.from_template(REWRITE_QUESTION_TEMPLATE)

# Answer Generation Prompt


SYSTEM_TEMPLATE = """
You are a helpful assistant that helps answer questions. Use only what is in the context: the question and the conversation history, with information that comes from your knowledge base. You need to decide whether to search for more context or answer directly. Always consult the base if the question is about internal policies of a company, labor laws, or information about the services and products offered by the company you represent. If you still don't know the answer after searching multiple times, just say you don't know. Respond in no more than three sentences and use the language of the question, directly and clearly. The question is: {question}."""
SYSTEM_PROMPT_TEMPLATE = PromptTemplate.from_template(SYSTEM_TEMPLATE)