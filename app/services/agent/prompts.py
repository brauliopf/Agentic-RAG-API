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

# Answer Generation Prompt


AGENT_TEMPLATE = """
You are a helpful assistant that helps answer questions based on the context provided: the question and the conversation history, with information from your knowledge base. You need to decide whether to access new information using a tool or to answer directly. If you still don't know the answer and are unsure about what to do, just say you don't know and ask the user to provide more information. Respond in no more than three sentences, in the same language of the question. Respond directly and clearly. The question is: {question}."""
AGENT_PROMPT_TEMPLATE = PromptTemplate.from_template(AGENT_TEMPLATE)