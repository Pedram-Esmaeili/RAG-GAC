import os
import asyncio
import logging

from dotenv import load_dotenv

from ..utils.socket_manager import socket_manager
from .rag_service import RAGService # Import RAGService

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

class RagChatbotService:
    """
    A service for chat interactions using the RAGService.
    This implementation focuses on RAG queries and uses the socket manager for communication.
    """

    _responses_to_cancel = set()  # Set of user_ids whose responses should be cancelled

    @classmethod
    def cancel_response(cls, user_id):
        """Mark a user's response as cancelled"""
        cls._responses_to_cancel.add(user_id)
        return True

    @classmethod
    def should_cancel_response(cls, user_id):
        """Check if a user's response should be cancelled"""
        return user_id in cls._responses_to_cancel

    @classmethod
    def clear_response_cancellation(cls, user_id):
        """Clear the cancellation flag for a user"""
        cls._responses_to_cancel.discard(user_id)

    @staticmethod
    async def get_response(
        user_messages,
        voice=False,
        user_id='ss',
        connection_id=''
        ):
        """
        Get a response from the RAG model.
        """
        complete_response = ""
        if voice:
         await socket_manager.send_status(connection_id=connection_id, status="start",text_=False,voice=True)
        else:
         await socket_manager.send_status(connection_id=connection_id, status="start",text_=True,voice=False)   
        try:
            # Clear any previous cancellation flag
            RagChatbotService.clear_response_cancellation(user_id)

            # Initialize RAGService
            rag_service = RAGService()

            # Extract the latest user message
            logger.info(f"RAG Chat request - User ID: {user_id}, Message: {user_messages}")

            # Get response from RAGService
            result = await rag_service.ask_async(user_messages, voice=voice,connection_id=connection_id)
            complete_response = result.get("answer", "")

            # Log latency and token counts in a formatted way
            total_latency = result.get("latency_s", 0.0)
            generation_latency = result.get("generation_latency_s", 0.0)
            prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
            total_tokens = result.get("usage", {}).get("total_tokens", 0)
            from_cache = result.get("from_cache", False)

            logger.info(f"\n--- RAG Chat Response Summary ---")
            logger.info(f"User ID: {user_id}")
            logger.info(f"From Cache: {from_cache}")
            logger.info(f"Total Latency: {total_latency:.2f}s")
            logger.info(f"Generation Latency: {generation_latency:.2f}s")
            logger.info(f"Prompt Tokens: {prompt_tokens}")
            logger.info(f"Completion Tokens: {completion_tokens}")
            logger.info(f"Total Tokens: {total_tokens}")
            logger.info(f"-----------------------------------\n")

            # Final message to client via socket
            await socket_manager.send_to_user(connection_id,user_id, {
                "type": "message",
                "messages": complete_response
            })
            await socket_manager.send_message(connection_id, {
                "type": "text_message",
                "status": "complete"
            })

            
        except asyncio.CancelledError:
            logger.info(f"RAG Chat process cancelled for user {user_id} on connection {connection_id}")
            await socket_manager.send_message(connection_id, {
                "type": "response_cancelled",
                "message": "Response was cancelled by user request"
            })
            raise
        except Exception as e:
            logger.error(f"Error processing RAG chat request: {str(e)}")
            await socket_manager.send_message(connection_id, {
                "type": "error",
                "user_id": user_id,
                "message": f"Error: {str(e)}"
            })
        finally:
            RagChatbotService.clear_response_cancellation(user_id)
            logger.info(f"RAG Chat response, User ID: {user_id}, Message: {complete_response}")

        # Return the full structured result so callers can use usage/latencies/contexts
        return result
