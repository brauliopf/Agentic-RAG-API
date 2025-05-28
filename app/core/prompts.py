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
Look at the input and try to reason about the underlying semantic intent / meaning.
Here is the initial question:
------- 
{question}
------- 
Formulate an improved question:
"""
REWRITE_QUESTION_PROMPT_TEMPLATE = PromptTemplate.from_template(REWRITE_QUESTION_TEMPLATE)

# Answer Generation Prompt
GENERATE_ANSWER_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context}
"""
GENERATE_ANSWER_PROMPT_TEMPLATE = PromptTemplate.from_template(GENERATE_ANSWER_TEMPLATE) 

SYSTEM_TEMPLATE = """
You are a friendly assistant for question-answering tasks. You reply strictly using the context provided: a question and a conversation history, with content retrieved. You will need to decide whether: to retrieve more context or to answer the question. If you don't know the answer even after retrieving the context once or multiple times, just say that you don't know. Use three sentences maximum and keep the answer concise. The question is: {question}."""
SYSTEM_PROMPT_TEMPLATE = PromptTemplate.from_template(SYSTEM_TEMPLATE)