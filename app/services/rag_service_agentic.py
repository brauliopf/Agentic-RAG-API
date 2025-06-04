import uuid
import time
import json

from langgraph.graph import START, StateGraph
from typing import Dict, Any, Optional, Literal
from langgraph.checkpoint.memory import MemorySaver

from ..core.logging import get_logger
from ..core.prompts import GRADE_DOCUMENTS_TEMPLATE, REWRITE_QUESTION_TEMPLATE, SYSTEM_PROMPT_TEMPLATE
from .llm_service import llm_service
from .document_service import document_service
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState, StateGraph
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import StateGraph, START, END
from ..models.requests import GradeDocuments
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


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
            self.graph = self._build_graph()
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

        # 1. Create Retriever Tool
        retriever = document_service.vector_store.as_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_knowledge_base",
            "Search and return information from a knowledge base.",
        )

        # 2. Get graph builder
        workflow = StateGraph(MessagesState)

        # 3. Node to decide if we need to retrieve or respond
        # If we need to retrieve, "response" will contain a tool call
        # If we don't need to retrieve, "response" will contain a message with the answer
        def generate_query_or_respond(state: MessagesState):
            """Decide to retrieve, or simply respond to the user.
            Takes all the messages in the state and returns a response."""

            last_human_message = state["messages"][-1]
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(question=last_human_message.content)

            if len(state["messages"]) >= 5:
                # Invoke the model to summarize the conversation
                summary_prompt = (
                    "Distill the above chat messages into a single summary message. "
                    "Include as many specific details as you can."
                )
                summary_message = llm_service.llm.invoke(
                    state["messages"] + [HumanMessage(content=summary_prompt)]
                )

                # Delete messages that we no longer want to show up (default to entire conversation history)
                delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]

                # Generate response based on the summary of the conversation
                response = llm_service.llm.invoke(
                    [summary_message, last_human_message, SystemMessage(content=sys_prompt)]
                )
                
                # Return both the response and delete messages as separate items in the messages list
                return {"messages": [summary_message, last_human_message, response] + delete_messages}
            else:
                messages = [*state["messages"][:-1], SystemMessage(content=sys_prompt)]
                response = llm_service.llm.bind_tools([retriever_tool]).invoke(messages)

            return {"messages": [response]}
        
        workflow.add_node(generate_query_or_respond)
        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_node("retriever_node", ToolNode([retriever_tool]))
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition, # LangGraph's function to check if the LLM's response contains tool calls
            {
                "tools": "retriever_node", # if there is tool call, go to the tools node
                END: END, # if there is not tool call, go to the end node
            },
        )

        # 4. Node to rewrite the question
        def rewrite_question(state: MessagesState):
            """Rewrite the original user question."""
            old_question = state["messages"][0].content

            # Get the last human input before the retriever_node
            question = None
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    question = message.content
                    break

            print("\nREWRITE QUESTION!!!\n")
            prompt = REWRITE_QUESTION_TEMPLATE.format(question=question)
            response = llm_service.llm.invoke([HumanMessage(content=prompt)])
            print("\n\nResponse!!! -> ", response)
            response_json = json.loads(response.content)
            
            # Add the rewritten question to the existing messages
            rewritten_question = HumanMessage(content=response_json["question"])
            return {"messages": [rewritten_question]}
        workflow.add_node(rewrite_question)

        def generate_answer(state: MessagesState):
            """Generate an answer."""

            # Get the last human input before the retriever_node
            question = None
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    question = message.content
                    break
            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(question=question)
            messages = state["messages"] + [SystemMessage(content=sys_prompt)]
            response = llm_service.llm.invoke(messages)


            return {"messages": [response]}
        workflow.add_node(generate_answer)

        # conditional edge from tools_node to generate_answer or rewrite_question
        grader_model = llm_service.llm
        def grade_documents(
            state: MessagesState,
        ) -> Literal["generate_answer", "rewrite_question"]:
            """Determine whether the retrieved documents are relevant to the question."""
            oldquestion = state["messages"][0].content
            
            # Get the last human input before the retriever_node
            question = None
            for message in reversed(state["messages"]):
                if isinstance(message, HumanMessage):
                    question = message.content
                    break
            
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
        question: str,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a RAG query and return the result."""
        query_id = str(uuid.uuid4())
        start_time = time.time()

        if thread_id is None:
            thread_id = f"query_{query_id}"
        
        try:
            logger.info("Starting RAG query", query_id=query_id, question=question, thread_id=thread_id)
            
            # Create config with thread_id for session persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Add the new user message to the conversation
            input_message = {"messages": [HumanMessage(content=question)]}
            
            # Run the RAG pipeline
            # Graph has a checkpointer that will maintain conversation history via the thread_id
            result = await self.graph.ainvoke(input_message, config=config)
            
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