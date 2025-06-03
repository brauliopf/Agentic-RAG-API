from .services.rag_service_agentic import rag_service_agentic
from .services.document_service import document_service
from .models.requests import QueryRequest
from typing import Optional
import asyncio
import gradio as gr

async def query(request: QueryRequest):
    return await rag_service_agentic.query(request.question, request.thread_id)

def main(input, thread_id: Optional[str] = None):
    request = QueryRequest(question=input, thread_id=thread_id)
    result = asyncio.run(query(request))
    return result["answer"]

demo = gr.Interface(
    fn=main,
    inputs=["text", "text"],
    outputs=["text"],
)

if __name__ == "__main__":
    docs = [
        ("https://www.usetako.com/termos-de-uso","tako"),
        ("https://www.usetako.com/politica-de-privacidade","tako"),
    ]
    for doc in docs:
        asyncio.run(document_service.ingest_url(url=doc[0]))

    files = [
        # ("/Users/brauliopf/Documents/Dev/langchain/docs/CLT Normas Correlatas 6th Ed.pdf","labor_rules")
        # ("/Users/brauliopf/Documents/Dev/langchain/docs/test.pdf", "test"),
        # ("/Users/brauliopf/Documents/Dev/langchain/docs/internal-policies.md", "internal_policies"),
        # ("/Users/brauliopf/Documents/Dev/langchain/docs/test.pdf", "tako"),
    ]
    for file in files:
        asyncio.run(document_service.ingest_file(file_path=file[0]))
    
    port = 7860
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)