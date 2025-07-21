from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from copy import deepcopy
from .tools import retrieve_for_user_id, tavily_search_tool, query_video, read_image
from ..llm_service import llm_service
from .prompts import AGENT_PROMPT_TEMPLATE
from .schemas import UserMessagesState
import json
from langgraph.prebuilt import ToolNode


tools = [tavily_search_tool, query_video, read_image]
tools_node = ToolNode(tools)


def should_retrieve(state: UserMessagesState):
    question = get_last_human_message(state)
    # Output a tool call, not the result
    tool_call = {
        "name": "retrieve_for_user_id",
        "args": {
            "user_id": state["user_id"],
            "query": question
        },
        "id": "retrieve_tool_call"
    }
    ai_message = AIMessage(
        content="Calling retrieve_for_user_id tool.",
        tool_calls=[tool_call]
    )
    return {"messages": [ai_message]}

async def retriever_node(state: UserMessagesState):
    """Retrieve information from the knowledge base."""
    # Get the last human input before the retriever_node
    question = get_last_human_message(state)
    
    # Use invoke instead of direct call and await the async function
    response = await retrieve_for_user_id.ainvoke({"user_id": state["user_id"], "query": question})

    # Create a proper ToolMessage with the tool name
    tool_message = ToolMessage(
        content=response,
        tool_call_id="retrieve_tool_call", 
        name="retrieve_for_user_id"
    )

    return {"messages": [tool_message]}

def agent_node(state: UserMessagesState):
    """Decide to retrieve, or simply respond to the user.
    Takes all the messages in the state and returns a response."""

    # Get all messages, enhancing the last message with the system prompt
    last_message = state["messages"][-1] # either the first query or a rewritten query
    agent_prompt = AGENT_PROMPT_TEMPLATE.format(question=last_message.content)
    messages = [*state["messages"][:-1], SystemMessage(content=agent_prompt)]

    # Bind tools to the LLM
    tools.pop(0)
    print(tools)
    llm_with_tools = llm_service.llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def get_last_human_message(state: UserMessagesState):
    """Get the last human message from the state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return None

def generate_answer(state: UserMessagesState):
    """Generate an answer."""

    # Get the last human input before the retriever_node
    question = get_last_human_message(state)
    sys_prompt = AGENT_PROMPT_TEMPLATE.format(question=question)
    messages = state["messages"] + [SystemMessage(content=sys_prompt)]
    response = llm_service.llm.invoke(messages)

    return {"messages": [response]}  # Wrap response in a list

######################################
############ BACKUP NODES ############
######################################

# Question Rewriting Prompt
from langchain_core.prompts import PromptTemplate

REWRITE_QUESTION_TEMPLATE = """
You are a helpful assistant that helps answer questions. You rewrite the question received to try to extract additional information from your database, to inform yourself and answer the question in a more complete way. This is the question received:
------- 
{question}
-------
Edit, but do not alter the meaning or intention of a question. Respond to this message in a structured format, with an object with a key "question" and the value set to the edited question. Do not include any other text or formatting.
For example: Ex: {{"question": "IMPROVED QUESTION?"}}
"""
REWRITE_QUESTION_PROMPT_TEMPLATE = PromptTemplate.from_template(REWRITE_QUESTION_TEMPLATE)

def rewrite_question(state: UserMessagesState):
    """Rewrite the original user question."""
    # Get the last human input before the retriever_node
    question = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            question = message.content
            break

    prompt = REWRITE_QUESTION_TEMPLATE.format(question=question)
    response = llm_service.llm.invoke([HumanMessage(content=prompt)])
    response_json = json.loads(response.content)
    
    # Add the rewritten question to the existing messages as a HumanMessage
    rewritten_question = HumanMessage(content=response_json["question"])
    return {"messages": [rewritten_question]}