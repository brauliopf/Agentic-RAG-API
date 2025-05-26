import uuid
import time
from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict, Annotated
from datetime import datetime

from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

from ..core.config import settings
from ..core.logging import get_logger
from .llm_service import llm_service
from .document_service import document_service
from langchain_core.prompts import PromptTemplate
from langgraph.graph import MessagesState, StateGraph
from langchain.tools.retriever import create_retriever_tool

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from ..models.requests import GradeDocuments
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


logger = get_logger(__name__)

# Define the search query structure for query analysis
# Used in the "non-agentic" pipeline
class Search(TypedDict):
    """Search query structure for query analysis."""
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# Define the state of the "non-agentic" RAG pipeline
class State(TypedDict):
    """RAG pipeline state."""
    question: str
    query: Search
    context: List[Document]
    answer: str


class NonAgenticRAGService:
    """Service for handling RAG queries using LangGraph pipeline."""
    
    def __init__(self):
        self.queries: Dict[str, Dict[str, Any]] = {}
        self.prompt = None
        self.graph = None
        self._initialize_pipeline()
        logger.info("Initialized RAGService nonagentic")
    
    def _initialize_pipeline(self):
        """Initialize the RAG pipeline with LangGraph."""
        try:
            custom_prompt_template = """
            You are an assistant for question-answering tasks. You reply strictly using the context provided. Use the following pieces of retrieved context to answer the question. If the context provided does not contain the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:
            """
            custom_prompt = PromptTemplate.from_template(custom_prompt_template)
            self.prompt = custom_prompt
            
            # Build the graph
            self.graph = self._build_graph()
            logger.info("RAG pipeline initialized successfully")
            # Save graph visualization to file
            graph_png = self.graph.get_graph().draw_mermaid_png()
            with open("./app/services/graph_nonagentic.png", "wb") as f:
                f.write(graph_png)
            
        except Exception as e:
            logger.error("Failed to initialize RAG pipeline", error=str(e))
            raise
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph pipeline."""
        def analyze_query(state: State):
            """Analyze the query to extract search parameters."""
            try:
                structured_llm = llm_service.llm.with_structured_output(Search)
                query = structured_llm.invoke(state["question"])
                return {"query": query}
            except Exception as e:
                logger.error("Query analysis failed", error=str(e))
                # Fallback to simple query structure
                return {
                    "query": {
                        "query": state["question"],
                        "section": None
                    }
                }
        
        def retrieve(state: State):
            """Retrieve relevant documents."""
            try:
                query = state["query"]
                
                # Build filter function if section is specified
                filter_func = None
                if query.get("section"):
                    filter_func = lambda doc: doc.metadata.get("section") == query["section"]
                
                # Retrieve documents
                retrieved_docs = document_service.vector_store.similarity_search(
                    query["query"],
                    k=settings.max_docs_retrieval,
                    filter=filter_func
                )
                
                return {"context": retrieved_docs}
                
            except Exception as e:
                logger.error("Document retrieval failed", error=str(e))
                return {"context": []}
        
        def generate(state: State):
            """Generate answer from context."""
            try:
                docs_content = "\n\n".join(doc.page_content for doc in state["context"])
                messages = self.prompt.invoke({
                    "question": state["question"], 
                    "context": docs_content
                })
                response = llm_service.llm.invoke(messages)
                return {"answer": response.content}
                
            except Exception as e:
                logger.error("Answer generation failed", error=str(e))
                return {"answer": "I apologize, but I encountered an error while generating the answer."}
        
        # Build and compile the graph
        graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
        graph_builder.add_edge(START, "analyze_query")
        return graph_builder.compile()
    
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
            
            # Store query metadata
            self.queries[query_id] = {
                "id": query_id,
                "question": question,
                "status": "processing",
                "created_at": datetime.utcnow(),
                "max_docs": max_docs or settings.max_docs_retrieval,
                "section_filter": section_filter
            }
            
            # Prepare initial state
            initial_state = {
                "question": question,
                "query": {"query": question, "section": section_filter},
                "context": [],
                "answer": ""
            }
            
            # Run the RAG pipeline
            result = self.graph.invoke(initial_state)
            
            processing_time = time.time() - start_time
            
            # Prepare context for response
            context = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["context"]
            ]
            
            # Update query status
            self.queries[query_id].update({
                "status": "completed",
                "answer": result["answer"],
                "context": context,
                "processing_time": processing_time
            })
            
            logger.info(
                "RAG query completed", 
                query_id=query_id, 
                processing_time=processing_time,
                context_docs=len(context)
            )
            
            return {
                "id": query_id,
                "question": question,
                "answer": result["answer"],
                "context": context,
                "processing_time": processing_time,
                "created_at": self.queries[query_id]["created_at"]
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error("RAG query failed", query_id=query_id, error=str(e))
            
            if query_id in self.queries:
                self.queries[query_id].update({
                    "status": "failed",
                    "error": str(e),
                    "processing_time": processing_time
                })
            
            raise

class AgenticRAGService:
    """
    Service for handling RAG queries using LangGraph agentic pipeline.
    Graph components operate on the "MessagesState", provided by LangGraph.
    """

    def __init__(self):
        self.prompt = None # the original prompt
        self.graph = None # the graph
        self._initialize_pipeline()
        logger.info("Initialized Agentic RAGS ervice")
    
    def _initialize_pipeline(self):
        """Initialize the Agentic RAG pipeline with LangGraph."""
        try:
            custom_prompt_template = """
            You are an assistant for question-answering tasks. You reply strictly using the context provided. Use the following pieces of retrieved context to answer the question. If the context provided does not contain the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:
            """
            custom_prompt = PromptTemplate.from_template(custom_prompt_template)
            self.prompt = custom_prompt
            
            # Build the graph
            self.graph = self._build_graph()
            logger.info("Agentic RAG pipeline initialized successfully")
            # Save graph visualization to file
            graph_png = self.graph.get_graph().draw_mermaid_png()
            with open("./app/services/graph_agentic.png", "wb") as f:
                f.write(graph_png)
            
        except Exception as e:
            logger.error("Failed to initialize RAG pipeline", error=str(e))
            raise
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph pipeline."""

        retriever = document_service.vector_store.as_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_blog_posts",
            "Search and return information about Lilian Weng blog posts.",
        )

        def generate_query_or_respond(state: MessagesState):
            """
            Use llm to generate a response based on the current state. Given the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
            """

            response = (llm_service.llm.bind_tools([retriever_tool]).invoke(state["messages"]))
            return {"messages": [response]}

        GRADE_PROMPT = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        )


        grader_model = llm_service.llm


        def grade_documents(
            state: MessagesState,
        ) -> Literal["generate_answer", "rewrite_question"]:
            """Determine whether the retrieved documents are relevant to the question."""
            question = state["messages"][0].content
            context = state["messages"][-1].content

            prompt = GRADE_PROMPT.format(question=question, context=context)
            response = (
                grader_model
                .with_structured_output(GradeDocuments).invoke(
                    [{"role": "user", "content": prompt}]
                )
            )
            score = response.binary_score

            if score == "yes":
                return "generate_answer"
            else:
                return "rewrite_question"

        REWRITE_PROMPT = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "{question}"
            "\n ------- \n"
            "Formulate an improved question:"
        )


        def rewrite_question(state: MessagesState):
            """Rewrite the original user question."""
            messages = state["messages"]
            question = messages[0].content
            prompt = REWRITE_PROMPT.format(question=question)
            response = llm_service.llm.invoke([{"role": "user", "content": prompt}])
            return {"messages": [{"role": "user", "content": response.content}]}
        

        GENERATE_PROMPT = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \n"
            "Context: {context}"
        )


        def generate_answer(state: MessagesState):
            """Generate an answer."""
            question = state["messages"][0].content
            context = state["messages"][-1].content
            prompt = GENERATE_PROMPT.format(question=question, context=context)
            response = llm_service.llm.invoke([{"role": "user", "content": prompt}])
            return {"messages": [response]}
        

        # ASSEMBLE THE GRAPH
        workflow = StateGraph(MessagesState)

        # Define the nodes we will cycle between
        workflow.add_node(generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([retriever_tool]))
        workflow.add_node(rewrite_question)
        workflow.add_node(generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            # Assess LLM decision (call `retriever_tool` tool or respond to the user)
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            grade_documents,
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        # Compile
        return workflow.compile()
    
    # async def query(
    #     self, 
    #     question: str, 
    #     max_docs: Optional[int] = None,
    #     section_filter: Optional[Literal["beginning", "middle", "end"]] = None
    # ) -> Dict[str, Any]:
    #     """Process a RAG query and return the result."""
    #     query_id = str(uuid.uuid4())
    #     start_time = time.time()
        
    #     try:
    #         logger.info("Starting RAG query", query_id=query_id, question=question)
            
    #         # Store query metadata
    #         self.queries[query_id] = {
    #             "id": query_id,
    #             "question": question,
    #             "status": "processing",
    #             "created_at": datetime.utcnow(),
    #             "max_docs": max_docs or settings.max_docs_retrieval,
    #             "section_filter": section_filter
    #         }
            
    #         # Prepare initial state
    #         initial_state = {
    #             "question": question,
    #             "query": {"query": question, "section": section_filter},
    #             "context": [],
    #             "answer": ""
    #         }
            
    #         # Run the RAG pipeline
    #         result = self.graph.invoke(initial_state)
            
    #         processing_time = time.time() - start_time
            
    #         # Prepare context for response
    #         context = [
    #             {
    #                 "content": doc.page_content,
    #                 "metadata": doc.metadata
    #             }
    #             for doc in result["context"]
    #         ]
            
    #         # Update query status
    #         self.queries[query_id].update({
    #             "status": "completed",
    #             "answer": result["answer"],
    #             "context": context,
    #             "processing_time": processing_time
    #         })
            
    #         logger.info(
    #             "RAG query completed", 
    #             query_id=query_id, 
    #             processing_time=processing_time,
    #             context_docs=len(context)
    #         )
            
    #         return {
    #             "id": query_id,
    #             "question": question,
    #             "answer": result["answer"],
    #             "context": context,
    #             "processing_time": processing_time,
    #             "created_at": self.queries[query_id]["created_at"]
    #         }
            
    #     except Exception as e:
    #         processing_time = time.time() - start_time
    #         logger.error("RAG query failed", query_id=query_id, error=str(e))
            
    #         if query_id in self.queries:
    #             self.queries[query_id].update({
    #                 "status": "failed",
    #                 "error": str(e),
    #                 "processing_time": processing_time
    #             })
            
    #         raise

# Global service instance
nonagentic_rag_service = NonAgenticRAGService() 
agentic_rag_service = AgenticRAGService() 