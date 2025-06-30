import uuid
import time
import json
import os
import asyncio

from langgraph.graph import START, StateGraph
from typing import Dict, Any, Optional, Literal
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated
from operator import add

from ..core.logging import get_logger
from ..core.prompts import GRADE_DOCUMENTS_TEMPLATE, REWRITE_QUESTION_TEMPLATE, SYSTEM_PROMPT_TEMPLATE
from .llm_service import llm_service
from .document_service import document_service
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage, BaseMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from ..models.requests import GradeDocuments
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from ..core.redis_client import redis

logger = get_logger(__name__)

# Custom state that extends MessagesState to include user_id
class UserMessagesState(TypedDict):
    """Extended MessagesState with user_id for user-specific operations."""
    messages: Annotated[list[BaseMessage], add]
    sources: Annotated[list[str], add]
    user_id: str

class RAGServiceAgentic:
    """
    Service for handling RAG queries using a LangGraph.
    """

    def __init__(self):
        try:
            # Build the graph
            # ** This graph trusts the retrieval process more than it trusts the LLM judgement about the use of the tool.
            # If the doc grader says the content is good, then the next node does not allow for retrieval, it's got to answer!
            # Thus, the quality of the grader model is critical.
            self.graph = self._build_graph()
            self._save_graph()
            
        except Exception as e:
            logger.error("Failed to initialize RAG pipeline", error=str(e))
            raise
        logger.info("Initialized Agentic RAG Service")
    
    def _save_graph(self):
        # Save graph visualization to file
        graph_png = self.graph.get_graph().draw_mermaid_png()
        # Use absolute path based on current file location
        # current_dir gets the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(current_dir, "graphs", "graph_agentic.png")
        with open(graph_path, "wb") as f:
            f.write(graph_png)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph pipeline."""

        # Create a user-specific retriever function that will be called with state context
        async def async_similarity_search(vector_store, query, filter):
            loop = asyncio.get_event_loop()
            logger.info("Run Similarity search", query=query, filter=filter)
            return await loop.run_in_executor(
                # reference langchain + pinecone: 
                None, lambda: vector_store.similarity_search_with_score(query, k=4, filter=filter)
            )

        # Fetch doc_group_ids from Redis for the user
        def get_doc_group_ids(user_id: str) -> list[str]:
            value = redis.get(user_id)
            if value:
                return [""] + value.split(",")
            return []

        async def retrieve_for_user_parallel(state, query):
            user_id = state.get("user_id", "_")
            
            # Get group ids from Redis
            doc_groups = get_doc_group_ids(user_id)
            if not doc_groups:
                doc_groups = []
            logger.info("Get list of curated doc groups", user_id=user_id, doc_groups=doc_groups)

            tasks = []
            # append first task to query default index (activated only)
            default_vector_store = document_service._get_vector_store_with_namespace()
            for group in doc_groups:
                filter = {"doc_group": group}
                tasks.append(async_similarity_search(default_vector_store, query, filter))
            
            # append second task to query user index
            user_vector_store = document_service._get_vector_store_with_namespace(user_id)
            tasks.append(async_similarity_search(user_vector_store, query, {}))

            # Get results from all parallel tasks (wait for all to complete)
            # Each task retrieves from a doc_group_id and returns up to 4 results (k=4)
            # If no match is found, the task returns an empty list
            # Each result has the following structure: (doc{id, metadata, page_content}, similarity_score)
            # Flatten. Sort docs. Take top 5,
            results = await asyncio.gather(*tasks)
            all_docs = [doc for group in results for doc in group]
            top_docs = sorted(all_docs, key=lambda x: x[1], reverse=True)[:5]
            logger.info("Top docs", top_docs=top_docs)
            
            # Format results - Extract and serialize the document objects
            retrieved_docs = [doc for doc, _ in top_docs]
            serialized = "\n\n".join(
                f"Source: {doc.metadata}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )
            
            logger.info("Retrieved documents", user_id=user_id, num_docs=len(retrieved_docs))
            return serialized

        # Placeholder for a custom retriever tool (ref: custom_retriever_node)
        # Declare the tool name to let the agent invoke the tool (this is a placeholder only).
        @tool
        def retrieve_knowledge_base(query: str) -> str:
            """Search and return information from a user-specific knowledge base."""
            return query
        
        retriever_tool = retrieve_knowledge_base

        # Custom tool node that can access state
        # This is the actual function that delivers the retrieved documents to the agent.
        async def custom_retriever_node(state: UserMessagesState):
            """Custom tool node that handles retrieval with user context."""
            # Get the last message with tool calls
            last_tool_request_message = None
            for message in reversed(state["messages"]):
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    last_tool_request_message = message
                    break
            
            if not last_tool_request_message or not last_tool_request_message.tool_calls:
                return {"messages": []}
            
            # Process tool calls
            tool_messages = []
            for tool_call in last_tool_request_message.tool_calls:
                if tool_call["name"] == "retrieve_knowledge_base":
                    query = tool_call["args"]["query"]
                    result = await retrieve_for_user_parallel(state, query)
                    
                    # Create tool message
                    tool_message = ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                    tool_messages.append(tool_message)
            
            return {"messages": tool_messages}

        # 2. Get graph builder with custom state
        workflow = StateGraph(UserMessagesState)

        # 3. Node to decide if we need to retrieve or respond
        # If we need to retrieve, "response" will contain a tool call
        # If we don't need to retrieve, "response" will contain a message with the answer
        def generate_query_or_respond(state: UserMessagesState):
            """Decide to retrieve, or simply respond to the user.
            Takes all the messages in the state and returns a response."""

            last_message = state["messages"][-1] # either the first query or a rewritten query
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(question=last_message.content)

            chat_length = sum(len(m.content) for m in state['messages'])

            if chat_length >= 10000000:
                # Invoke the model to summarize the conversation
                summary_prompt = (
                    "Distill the above chat messages into a single summary message. "
                    "Include as many specific details as you can."
                    "Include a summmary of every question and answer exchange, structured as an ordered list of questions (sender and content) and answers (sender and content)."
                )
                summary_message = llm_service.llm.invoke(
                    state["messages"] + [HumanMessage(content=summary_prompt)]
                )

                # Delete messages that we no longer want to show (default to entire conversation history)
                delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]

                # Generate response based on the summary of the conversation
                response = llm_service.llm.invoke(
                    [summary_message, last_message, SystemMessage(content=sys_prompt)]
                )
                
                # Return both the response and delete messages as separate items in the messages list
                return {"messages": [summary_message, last_message, response] + delete_messages}
            else:
                messages = [*state["messages"][:-1], SystemMessage(content=sys_prompt)]
                response = llm_service.llm.bind_tools([retriever_tool]).invoke(messages)

            return {"messages": [response]}
        
        workflow.add_node(generate_query_or_respond)
        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_node("retriever_node", custom_retriever_node)
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition, # LangGraph's function to check if the LLM's response contains tool calls
            {
                "tools": "retriever_node", # if there is tool call, go to the tools node
                END: END, # if there is not tool call, go to the end node
            },
        )

        def _get_last_human_message(state: UserMessagesState):
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
        workflow.add_node(rewrite_question)

        def generate_answer(state: UserMessagesState):
            """Generate an answer."""

            # Get the last human input before the retriever_node
            question = _get_last_human_message(state)
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(question=question)
            messages = state["messages"] + [SystemMessage(content=sys_prompt)]
            response = llm_service.llm.invoke(messages)

            return {"messages": [response]}  # Wrap response in a list
        workflow.add_node(generate_answer)

        # conditional edge from tools_node to generate_answer or rewrite_question
        grader_model = llm_service.llm
        def grade_documents(
            state: UserMessagesState,
        ) -> Literal["generate_answer", "rewrite_question"]:
            """Determine whether the retrieved documents are relevant to the question."""
            # Get the last human input before the retriever_node
            question = _get_last_human_message(state)
            
            # Get the last message with tool calls
            context = state["messages"][-1].content

            prompt = GRADE_DOCUMENTS_TEMPLATE.format(question=question, context=context)
            response = (
                grader_model
                .with_structured_output(GradeDocuments).invoke(
                    [HumanMessage(content=prompt)]
                )
            )

            if response.binary_score == "yes":
                return "generate_answer"
            else:
                return "rewrite_question"
            
        workflow.add_conditional_edges(
            "retriever_node",
            grade_documents
        )
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        workflow.add_edge("generate_answer", END)

        # Compile with memory checkpointer for session persistence
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    async def query(
        self, 
        query: str,
        thread_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Process a RAG query and return the result."""
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info("Starting RAG query", query_id=query_id, query=query, thread_id=thread_id, user_id=user_id)
            
            # Create config with thread_id for session persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Add the new user message to the conversation with user_id in state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "user_id": user_id
            }
            
            # Run the RAG pipeline
            # Graph has a checkpointer that will maintain conversation history via the thread_id
            result = await self.graph.ainvoke(initial_state, config=config)
            
            processing_time = time.time() - start_time
            
            logger.info(
                "RAG query completed", 
                query_id=query_id,
                processing_time=processing_time,
                user_id=user_id
            )
            
            return {
                "id": query_id,
                "question": query,
                "answer": result["messages"][-1].content,
                "context": [],
                "processing_time": processing_time,
                "created_at": time.time()
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("RAG query failed", query_id=query_id, error=str(e), user_id=user_id)

            raise

    @staticmethod
    def _extract_sources_from_tool_message(message_content: str) -> list[str]:
        """Extracts sources from the retriever tool message content."""
        sources = []
        for line in message_content.splitlines():
            if line.startswith("Source: "):
                # Extract after 'Source: ' and before (optional) newline or 'Content:'
                src = line[len("Source: "):].strip()
                if src:
                    sources.append(src)
        return sources

# Global service instance
rag_service_agentic = RAGServiceAgentic() 