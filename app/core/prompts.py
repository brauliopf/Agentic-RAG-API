"""
Centralized prompt templates for the RAG application.
"""

from langchain_core.prompts import PromptTemplate

# RAG Question-Answering Prompt
RAG_QA_TEMPLATE = """
You are an assistant for question-answering tasks. You reply strictly using the context provided. Use the following pieces of retrieved context to answer the question. If the context provided does not contain the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Document Grading Prompt
GRADE_DOCUMENTS_TEMPLATE = """
You are a grader assessing relevance of a retrieved document to a user question. 
Here is the retrieved document: 

{context} 

Here is the user question: {question} 

If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

# Question Rewriting Prompt
REWRITE_QUESTION_TEMPLATE = """
Look at the input and try to reason about the underlying semantic intent / meaning.
Here is the initial question:
------- 
{question}
------- 
Formulate an improved question:
"""

# Answer Generation Prompt
GENERATE_ANSWER_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context}
"""

# Create prompt template instances
RAG_QA_PROMPT = PromptTemplate.from_template(RAG_QA_TEMPLATE)
GRADE_DOCUMENTS_PROMPT = PromptTemplate.from_template(GRADE_DOCUMENTS_TEMPLATE)
REWRITE_QUESTION_PROMPT = PromptTemplate.from_template(REWRITE_QUESTION_TEMPLATE)
GENERATE_ANSWER_PROMPT = PromptTemplate.from_template(GENERATE_ANSWER_TEMPLATE) 