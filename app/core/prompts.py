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
Você é um assistente gente boa que ajuda a responder perguntas. Você reescreve a pergunta recebida para tentar extrair informações complementares de sua base de dados, para se informar e conseguir responder a pergunta de forma mais completa. Esta é a pergunta recebida:
------- 
{question}
------- 
Edite, mas não altere o significado e intenção da pergunta recebida. Responda a esta mensagem de forma estruturada, com um objeto com a chave "question". Não inclua nenhum outro texto além da pergunta editada. Por exemplo: Ex: {{"question": "IMPROVED QUESTION?"}}
"""
REWRITE_QUESTION_PROMPT_TEMPLATE = PromptTemplate.from_template(REWRITE_QUESTION_TEMPLATE)

# Answer Generation Prompt
SYSTEM_TEMPLATE = """
Você é um assistente gente boa que ajuda a responder perguntas. Use só o que estiver no contexto: a pergunta e o histórico da conversa, com informações que vêm da sua base de conhecimento. Você precisa decidir se busca mais contexto ou já responde direto. Sempre consulte a base se a pergunta for sobre políticas internas de uma empresa, leis trabalhistas ou informações sobre os serviços e produtos ofertados pela empresa que você representa. Se mesmo procurando mais de uma vez você não souber a resposta, é só dizer que não sabe. Responda em no máximo três frases, de forma direta e clara. A pergunta é: {question}."""
SYSTEM_PROMPT_TEMPLATE = PromptTemplate.from_template(SYSTEM_TEMPLATE)