from .services.rag_service_agentic import rag_service_agentic
from .services.document_service import document_service
from typing import Optional, Literal
import asyncio
import gradio as gr

async def query(question: str, max_docs: Optional[int] = None, section_filter: Optional[Literal["beginning", "middle", "end"]] = None, thread_id: Optional[str] = None):
    return await rag_service_agentic.query(question, max_docs, section_filter, thread_id)

def main(input, thread_id: Optional[str] = None):
    result = asyncio.run(query(input, None, None, thread_id))
    return result["answer"]

demo = gr.Interface(
    fn=main,
    inputs=["text", "text"],
    outputs=["text"],
)

if __name__ == "__main__":
    docs = [
        "https://www.usetako.com/termos-de-uso",
        "https://www.usetako.com/politica-de-privacidade",
    ]
    for doc in docs:
        asyncio.run(document_service.ingest_url_tako(doc))
    
    port = 7860
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)