import uuid
import time
from typing import Literal

from langgraph.graph import START, StateGraph
from typing import List, Dict, Any, Optional, Literal

from ..core.logging import get_logger
from ..core.prompts import GRADE_DOCUMENTS_TEMPLATE, REWRITE_QUESTION_TEMPLATE, GENERATE_ANSWER_TEMPLATE
from .llm_service import llm_service
from .document_service import document_service
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import StateGraph, START, END
from ..models.requests import GradeDocuments
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


logger = get_logger(__name__)

class RAGServiceAgentic:
    """
    Service for handling RAG queries using LangGraph agentic pipeline.
    """

    def __init__(self):
        try:
            # Build the graph
            self.graph = self._build_graph()
            logger.info("Agentic RAG pipeline initialized successfully")
            # Save graph visualization to file
            graph_png = self.graph.get_graph().draw_mermaid_png()
            with open("./app/services/graphs/graph_agentic.png", "wb") as f:
                f.write(graph_png)
            
        except Exception as e:
            logger.error("Failed to initialize RAG pipeline", error=str(e))
            raise
        logger.info("Initialized Agentic RAG Service")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph pipeline."""

        retriever = document_service.vector_store.as_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_blog_posts",
            "Search and return information about Lilian Weng blog posts.",
        )

        # BUILD THE GRAPH
        workflow = StateGraph(MessagesState)

        def generate_query_or_respond(state: MessagesState):
            """Decide to retrieve, or simply respond to the user."""

            # ADD SYSTEM PROMPT
            system_prompt = """
            You are an assistant for question-answering tasks. You reply strictly using the context provided. If the context provided does not contain the answer, just say that you don't know. Use the following pieces of retrieved context to answer the question. Use three sentences maximum and keep the answer concise.
            """
            msg_with_prompt = [SystemMessage(content=system_prompt)] + state["messages"]
            response = (llm_service.llm.bind_tools([retriever_tool]).invoke(msg_with_prompt))
            return {"messages": [response]}
        
        workflow.add_node(generate_query_or_respond)
        workflow.add_edge(START, "generate_query_or_respond")


        workflow.add_node("tools_node", ToolNode([retriever_tool]))
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition, # LangGraph's function to check if the LLM's response contains tool calls
            {
                "tools": "tools_node", # if there is tool call, go to the tools node
                END: END, # if there is not tool call, go to the end node
            },
        )

        def rewrite_question(state: MessagesState):
            """Rewrite the original user question."""
            messages = state["messages"]
            question = messages[0].content
            prompt = REWRITE_QUESTION_TEMPLATE.format(question=question)
            response = llm_service.llm.invoke([{"role": "user", "content": prompt}])
            return {"messages": [HumanMessage(content=response.content)]}
        workflow.add_node(rewrite_question)

        def generate_answer(state: MessagesState):
            """Generate an answer."""
            question = state["messages"][0].content
            context = state["messages"][-1].content
            prompt = GENERATE_ANSWER_TEMPLATE.format(question=question, context=context)
            response = llm_service.llm.invoke([{"role": "user", "content": prompt}])
            return {"messages": [response]}
        workflow.add_node(generate_answer)

        # conditional edge from tools_node to generate_answer or rewrite_question
        grader_model = llm_service.llm
        def grade_documents(
            state: MessagesState,
        ) -> Literal["generate_answer", "rewrite_question"]:
            """Determine whether the retrieved documents are relevant to the question."""
            question = state["messages"][0].content
            context = state["messages"][-1].content

            prompt = GRADE_DOCUMENTS_TEMPLATE.format(question=question, context=context)
            response = (
                grader_model
                .with_structured_output(GradeDocuments).invoke(
                    [{"role": "user", "content": prompt}]
                )
            )

            if response.binary_score == "yes":
                return "generate_answer"
            else:
                return "rewrite_question"
            
        workflow.add_conditional_edges(
            "tools_node",
            grade_documents
        )
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        workflow.add_edge("generate_answer", END)

        # Compile
        return workflow.compile()
    
    async def query(
        self, 
        question: str, 
        max_docs: Optional[int] = None,
        section_filter: Optional[Literal["beginning", "middle", "end"]] = None
    ) -> Dict[str, Any]:
        """Process a RAG query and return the result."""
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info("Starting RAG query", query_id=query_id, question=question)
                    
            # Prepare initial state
            initial_state = {
                "messages": [HumanMessage(content=question)]
            }
            
            # Run the RAG pipeline
            result = self.graph.invoke(initial_state)
            
            processing_time = time.time() - start_time
            
            logger.info(
                "RAG query completed", 
                query_id=query_id,
                processing_time=processing_time,
            )
            
            return {
                "id": query_id,
                "question": question,
                "answer": result["messages"][-1].content,
                "context": [],
                "processing_time": processing_time,
                "created_at": time.time()
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("RAG query failed", query_id=query_id, error=str(e))

            raise

# Global service instance
rag_service_agentic = RAGServiceAgentic() 