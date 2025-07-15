import asyncio
from app.core.logging import get_logger
from app.core.redis_client import redis
from app.services.document_service import document_service

logger = get_logger(__name__)

# Fetch doc_group_ids from Redis for the user
def _get_doc_group_ids(user_id: str) -> list[str]:
    value = redis.get(user_id)
    if value:
        return [""] + value.split(",")
    return []

# Create a user-specific retriever function that will be called with state context
async def _async_similarity_search(vector_store, query, filter):
    loop = asyncio.get_event_loop()
    logger.info("Run Similarity search", query=query, filter=filter)
    return await loop.run_in_executor(
        # reference langchain + pinecone: 
        None, lambda: vector_store.similarity_search_with_score(query, k=4, filter=filter)
    )

async def retrieve_execute_parallel(user_id, query):
  # Get doc group ids from Redis
  # This is the list of doc groups that the user has access to
  doc_groups = _get_doc_group_ids(user_id)
  if not doc_groups:
      doc_groups = []
  logger.info("Get list of curated doc groups", user_id=user_id, doc_groups=doc_groups)

  tasks = []

  # query default index for activated content only (doc_group)
  default_vectordb_namespace = document_service.get_vector_store_with_namespace("default")
  for group in doc_groups:
      filter = {"doc_group": group}
      tasks.append(_async_similarity_search(default_vectordb_namespace, query, filter))

  # query user index for all content
  user_vectordb_namespace = document_service.get_vector_store_with_namespace(user_id)
  tasks.append(_async_similarity_search(user_vectordb_namespace, query, {}))

  # Get results from all parallel tasks (wait for all to complete)
  # Each task retrieves from a doc_group_id and returns up to 4 results (k=4)
  # If no match is found, the task returns an empty list
  # Each top_doc has the following structure: (doc{id, metadata, page_content}, similarity_score)
  # Flatten. Sort docs. Take top 5,
  results = await asyncio.gather(*tasks)
  all_docs = [doc for group in results for doc in group]
  top_docs = sorted(all_docs, key=lambda x: x[1], reverse=True)[:5]
  logger.info("Top docs", top_docs=top_docs)

  # Format results - Extract and serialize the document objects
  retrieved_docs = [doc for doc, score in top_docs]
  serialized = "\n\n".join(
      f"Source: {doc.metadata}\nContent: {doc.page_content}"
      for doc in retrieved_docs
  )

  logger.info("Retrieved documents", user_id=user_id, num_docs=len(retrieved_docs))
  return serialized