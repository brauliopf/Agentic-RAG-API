import uuid
import time
import json
import os

from langgraph.graph import START, END, StateGraph
from typing import Dict, Any, Literal
from langgraph.checkpoint.memory import MemorySaver

from ..core.logging import get_logger
from .agent.prompts import GRADE_DOCUMENTS_TEMPLATE
from .llm_service import llm_service
from .document_service import document_service
from langchain_core.messages import HumanMessage

from ..models.requests import GradeDocuments
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from .agent.tools import tavily_search_tool, retrieve_for_user_id
from .agent.nodes import generate_query_or_respond, rewrite_question, generate_answer, get_last_human_message
from .agent.schemas import UserMessagesState
from copy import deepcopy

logger = get_logger(__name__)

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
            self.graph = self.build_graph()
            self.save_graph()
            
        except Exception as e:
            logger.error("Failed to initialize RAG pipeline", error=str(e))
            raise
        logger.info("Initialized Agentic RAG Service")
    
    def save_graph(self):
        # Save graph visualization to file
        graph_png = self.graph.get_graph().draw_mermaid_png()
        # Use absolute path based on current file location
        # current_dir gets the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        graph_path = os.path.join(current_dir, "agent/graphs", "graph_agentic.png")
        with open(graph_path, "wb") as f:
            f.write(graph_png)

    def build_graph(self) -> StateGraph:
        """Build the LangGraph pipeline."""

        workflow = StateGraph(UserMessagesState)
        tools = ToolNode([retrieve_for_user_id, tavily_search_tool])

        def grade_documents_router(
            state: UserMessagesState,
        ) -> Literal["generate_answer", "rewrite_question"]:
            """Determine whether the retrieved documents are relevant to the question."""
            # Get the last human input before the retriever_node
            question = get_last_human_message(state)
            
            # Get the last message with tool calls
            context = state["messages"][-1].content

            prompt = GRADE_DOCUMENTS_TEMPLATE.format(question=question, context=context)
            response = (
                llm_service.llm
                .with_structured_output(GradeDocuments).invoke(
                    [HumanMessage(content=prompt)]
                )
            )

            if response.binary_score == "yes":
                return "generate_answer"
            else:
                return "rewrite_question"
        
        workflow.add_node(generate_query_or_respond)
        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_node("tools_node", tools)
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition, # LangGraph's function to check if the LLM's response contains tool calls
            {
                "tools": "tools_node", # if there is a tool call, go to the tools node
                END: END, # if there is not a tool call, go to the end node
            },
        )
        workflow.add_node(rewrite_question)
        workflow.add_node(generate_answer)
        
        workflow.add_conditional_edges(
            "tools_node",
            grade_documents_router
        )
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        workflow.add_edge("generate_answer", END)

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
            
            # Extract sources from tool messages
            sources = []
            for message in result["messages"]:
                # Check if it's a tool message from the retriever
                if hasattr(message, 'name') and message.name == "retrieve_for_user_id":
                    extracted_sources = self.extract_sources_from_tool_message(message.content)
                    sources.extend(extracted_sources)
            
            processing_time = time.time() - start_time
            
            logger.info(
                "RAG query completed", 
                query_id=query_id,
                processing_time=processing_time,
                user_id=user_id,
                sources_count=len(sources)
            )
            
            return {
                "id": query_id,
                "question": query,
                "answer": result["messages"][-1].content,
                "context": sources,
                "processing_time": processing_time,
                "created_at": time.time()
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("RAG query failed", query_id=query_id, error=str(e), user_id=user_id)

            raise

    @staticmethod
    def extract_sources_from_tool_message(message_content: str) -> list[dict]:
        """Extracts sources from the retriever tool message content."""
        import ast
        
        sources = []
        for line in message_content.splitlines():
            if line.startswith("Source: "):
                # Extract after 'Source: ' and before (optional) newline or 'Content:'
                src = line[len("Source: "):].strip()
                if src:
                    try:
                        # Parse the string representation of the dictionary back to a dict
                        source_dict = ast.literal_eval(src)
                        sources.append(source_dict)
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"Failed to parse source metadata: {src}", error=str(e))
                        # Fallback: create a simple dict with the raw string
                        sources.append({"raw_metadata": src})
        return sources

# Global service instance
rag_service_agentic = RAGServiceAgentic() 