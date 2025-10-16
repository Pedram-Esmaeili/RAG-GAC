from fastapi import APIRouter, HTTPException, status, Request, Depends, WebSocket, WebSocketDisconnect, Query
from typing import cast, Dict, Any
from ...services.rag_chatbot_service import RagChatbotService # Import the new RAG Chatbot Service
from ...services.voice_service import VoiceService
from ...utils.socket_manager import socket_manager
from .responses import  uuid_responses
import time
import asyncio
import uuid
import logging
from ...api.routes.schemas import QueryRequest # Import the new QueryRequest schema
from ...services.rag_service import RAGService

# Create router
chat_router = APIRouter(
    
    tags=["Chat"]
)
logger: logging.Logger = logging.getLogger(__name__)

# WebSocket routes (updated with your JWT validation)
@chat_router.websocket("/stream/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """
    Establish WebSocket connection for real-time communication
    """
    try:
        
        logger.info(f"try to connection: {str(connection_id)}")
        #is_connected = await socket_manager.is_connection_active(connection_id)
        await socket_manager.connect(websocket, connection_id)
        client_ip = websocket.client.host if websocket.client else "unknown"
        logger.info(f"connected: {str(connection_id)}")
    
        while True:
            data = await websocket.receive_text()
            try:
                import json
                query_ = json.loads(data)
                # Handle ping to keep connection alive
                if query_.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
               
                # Handle stop request
                elif query_.get("type") == "stop":
                    # Cancel all tasks for this connection
                    tasks_cancelled = await socket_manager.cancel_connection_tasks(connection_id)
                    
                    # Also cancel any ongoing ChatbotService responses for user_ids associated with this connection
                    for user_id, conn_id in list(socket_manager.user_sessions.items()):
                        if conn_id == connection_id:
                            RagChatbotService.cancel_response(user_id) # Cancel RAG Chatbot Service response
                    
                    # Send acknowledgment
                    await websocket.send_json({
                        "type": "stop_ack",
                        "status": "success",
                        "message": f"Stopped all processing. Cancelled {tasks_cancelled} tasks."
                    })
                    
                    # Log the stop request
                    logger.info(f"Stop request - IP: {client_ip}, Connection ID: {connection_id}")
               
                # Handle chat request via WebSocket
                elif query_.get("type") == "chat_request":
                    # Cancel any ongoing tasks first (like a soft stop)
                    await socket_manager.cancel_connection_tasks(connection_id)
                    
                    # Extract chat parameters from message
                    user_id = query_.get("user_id",'')
                    message = query_.get("query")
                    voice_enabled = query_.get("voice")
                    chat_id = query_.get("chat_id")
                    if not chat_id:
                        chat_id = str(uuid.uuid4())
                        await socket_manager.send_chat_id(connection_id,user_id, chat_id)
                    
                    logger.info(f"Chat request - IP: {client_ip}, User ID: {user_id}, Message: {message}, Voice: {voice_enabled}")
                   
                    # Create chat processing task
                    task = asyncio.create_task(process_websocket_chat(
                        user_id=user_id,
                        messages=message,
                        voice=voice_enabled,
                        connection_id=connection_id,
                        chat_id=chat_id
                    ))
                    
                    # Register the task with the socket manager
                    socket_manager.register_task(connection_id, task)
                   
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await socket_manager.disconnect(connection_id)

async def process_websocket_chat(
        user_id,
        messages, # Corrected parameter name
        voice,
        connection_id,
        chat_id
):
    """Process chat request received via WebSocket"""
    start_time = time.time() # Start timer
    try:
        # Process the chat request
        response = await RagChatbotService.get_response(
                user_messages=messages,
                voice=voice,
                user_id=user_id,
                connection_id=connection_id
               
        )
        end_time = time.time() # End timer
        latency_ms = (end_time - start_time) * 1000

        # Expect a structured dict from service
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        token_count = usage.get("total_tokens")
        logger.info(
            f"Chat processed for user {user_id}, Chat ID: {chat_id} - "
            f"Token Count: {token_count if token_count is not None else 'n/a'}, "
            f"Latency: {latency_ms:.2f}ms"
        )
            
    except asyncio.CancelledError:
        # Handle cancellation gracefully
        logger.info(f"Chat process cancelled for user {user_id} on connection {connection_id}")
        
    except Exception as e:
        logger.error(f"Error processing WebSocket chat: {str(e)}")
        # Only send error if the connection is still active
        await socket_manager.send_message(connection_id, {
            "type": "error",
            "user_id": user_id,
            "message": f"Error: {str(e)}"
        })


@chat_router.post("/query")
async def query_endpoint(query_request: QueryRequest, request: Request):
    """
    Accepts a natural language query and returns a response from the RAG chatbot.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Query request - IP: {client_ip}, User ID: {query_request.user_id}, Query: {query_request.query}")

    try:
        rag = RAGService()
        result = await rag.ask_async(query=query_request.query)
        # Return structured payload
        return {
            "user_id": query_request.user_id,
            "answer": result.get("answer", ""),
            "usage": result.get("usage", {}),
            "latency_s": result.get("latency_s", 0),
            "generation_latency_s": result.get("generation_latency_s", 0),
            "contexts": result.get("contexts", []),
            "from_cache": result.get("from_cache", False)
        }
    except Exception as e:
        logger.error(f"Error processing query for user {query_request.user_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
