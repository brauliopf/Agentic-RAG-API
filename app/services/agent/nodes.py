from langchain_core.messages import HumanMessage, SystemMessage
from copy import deepcopy
from .tools import retrieve_for_user_id, tavily_search_tool
from ..llm_service import llm_service
from .prompts import SYSTEM_PROMPT_TEMPLATE, REWRITE_QUESTION_TEMPLATE
from .schemas import UserMessagesState
import json

# 1: Take task + Decide whether to respond or retrieve.
# Node to decide if we need to retrieve or respond
# If we need to retrieve, "response" will contain a tool call
# If we don't need to retrieve, "response" will contain a message with the answer
def generate_query_or_respond(state: UserMessagesState):
    """Decide to retrieve, or simply respond to the user.
    Takes all the messages in the state and returns a response."""

    last_message = state["messages"][-1] # either the first query or a rewritten query
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(question=last_message.content)

    messages = [*state["messages"][:-1], SystemMessage(content=sys_prompt)]

    llm_with_tools = llm_service.llm.bind_tools([retrieve_for_user_id, tavily_search_tool])
    response = llm_with_tools.invoke(messages)
    
    # If the response contains tool calls, inject user_id into them
    if response.tool_calls:
        updated_tool_calls = []
        for tool_call in response.tool_calls:
            tool_call_copy = deepcopy(tool_call)
            tool_call_copy["args"]["user_id"] = state["user_id"]
            updated_tool_calls.append(tool_call_copy)
        response.tool_calls = updated_tool_calls

    return {"messages": [response]}

def get_last_human_message(state: UserMessagesState):
    """Get the last human message from the state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content
    return None

# 4. Node to rewrite the question
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

def generate_answer(state: UserMessagesState):
    """Generate an answer."""

    # Get the last human input before the retriever_node
    question = get_last_human_message(state)
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(question=question)
    messages = state["messages"] + [SystemMessage(content=sys_prompt)]
    response = llm_service.llm.invoke(messages)

    return {"messages": [response]}  # Wrap response in a list