from .services.rag_service_agentic import rag_service_agentic
from .services.document_service import document_service
from typing import Optional, Literal
import asyncio
import gradio as gr

async def query(question: str, max_docs: Optional[int] = None, thread_id: Optional[str] = None):
    return await rag_service_agentic.query(question, max_docs, thread_id)

def main(input, thread_id: Optional[str] = None):
    result = asyncio.run(query(input, None, thread_id))
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
        asyncio.run(document_service.ingest_url(url=doc[0], url_type=doc[1], namespace=doc[1]))

    files = [
        # ("/Users/brauliopf/Documents/Dev/langchain/docs/CLT Normas Correlatas 6th Ed.pdf","labor_rules")
        # ("/Users/brauliopf/Documents/Dev/langchain/docs/test.pdf", "test"),
        # ("/Users/brauliopf/Documents/Dev/langchain/docs/internal-policies.md", "internal_policies"),
        # ("/Users/brauliopf/Documents/Dev/langchain/docs/test.pdf", "tako"),
    ]
    for file in files:
        asyncio.run(document_service.ingest_file(file_path=file[0], namespace=file[1]))
    
    port = 7860
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)