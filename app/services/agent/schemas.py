from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from operator import add

class UserMessagesState(TypedDict):
    """Extended MessagesState with user_id for user-specific operations."""
    messages: Annotated[list[BaseMessage], add]
    sources: Annotated[list[str], add]
    user_id: str
    should_RAG: bool