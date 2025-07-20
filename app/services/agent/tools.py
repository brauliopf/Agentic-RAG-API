from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from langchain_tavily import TavilySearch
from .retriever import retrieve_execute_parallel
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

@tool
def tavily_search_tool(
    query: Annotated[str, 'The query to search Tavily for']
) -> str:
    """Perform a search on Tavily"""
    return TavilySearch(max_results=3).run(query)

@tool
async def retrieve_for_user_id(query: str, user_id: Annotated[str, InjectedToolArg]) -> str:
  """Search and return information from a user-specific knowledge base."""
  return await retrieve_execute_parallel(user_id, query)

def encode_image(image_path: str) -> str:
    """
    Encode an image to base64
    Args:
      image_path (str): the absolute file_path of the targeted image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@tool
def read_image(image_path: str) -> str:
    """
    Read an image and return the text description.
    Args:
      image_path (str): the state variable 'attachment' has the absolute file_path of the targeted image.
    """
    base64_image = encode_image(image_path)

    # Initialize ChatOpenAI model
    llm = ChatOpenAI(
        model="gpt-4o",  # Updated to use gpt-4o for vision capabilities
        api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Create message with image
    message = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image? Describe the image in detail. If this is a board game, read the current status of the board, but do not make an analysis at all"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            },
        ]
    )

    # Get completion
    response = llm.invoke([message])
    return response.content

@tool
def query_video(video_url: str, query: str) -> dict:
    """
    Query a video using native Gemini API for video analysis.
    Args:
      video_url (str): The URL of the YouTube video to query.
      query (str): The question to ask about the video.
    Returns:
      dict: A structured answer with the fields: 'answer', 'reasoning'. For example: {video_url: 'https://www.youtube.com/watch?v=1htKBjuUWec', query: 'What does the lady in purple say when she enters the room?'} => {'answer': 'The lady says "Ok. Let\'s do this."', 'reasoning': 'Several people enter the room, a women in a purple shirt says "Ok. Let\'s do this."'}
    """
    # Use native API for video processing
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Define the response schema
    response_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The factual answer to the question about the video"
            },
            "reasoning": {
                "type": "string", 
                "description": "The reasoning behind the answer"
            }
        },
        "required": ["answer", "reasoning"]
    }
    
    response = model.generate_content(
        [
            {
                "parts": [
                    {
                        "file_data": {
                            "file_uri": video_url
                        }
                    },
                    {
                        "text": f"""Think step-by-step and respond to the following question about the video content. Your answer must be factual and concise.
                        Question: {query}"""
                    }
                ]
            }
        ],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema
        )
    )
    
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {"answer": response.text, "reasoning": "JSON parsing failed, returning raw response"}