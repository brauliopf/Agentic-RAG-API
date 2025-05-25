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

logger = get_logger(__name__)


class Search(TypedDict):
    """Search query structure for query analysis."""
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]


class State(TypedDict):
    """RAG pipeline state."""
    question: str
    query: Search
    context: List[Document]
    answer: str


class RAGService:
    """Service for handling RAG queries using LangGraph pipeline."""
    
    def __init__(self):
        self.queries: Dict[str, Dict[str, Any]] = {}
        self.prompt = None
        self.graph = None
        self._initialize_pipeline()
        logger.info("Initialized RAGService")
    
    def _initialize_pipeline(self):
        """Initialize the RAG pipeline with LangGraph."""
        try:
            # Load RAG prompt from hub:
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

# Global service instance
rag_service = RAGService() 