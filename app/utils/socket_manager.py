from fastapi import WebSocket
from typing import Dict, List, Any, Optional, Set
import json
import asyncio
import base64
import logging

logger = logging.getLogger(__name__)

class SocketManager:
    """
    Manages WebSocket connections and communication with clients.
    """
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.active_tasks: Dict[str, Set[asyncio.Task]] = {}  # Track tasks by connection_id
        self.user_sessions: Dict[str, str] = {}  # Maps user_id to connection_id
        
    async def connect(self, websocket: WebSocket, connection_id: str):
        """
        Connect a new WebSocket client
        """
        try:
                websocket_old = self.active_connections[connection_id]
                await websocket_old.close(code=1001, reason="Cleanup")
                logger.info(f"disconnect old connection: {str(connection_id)}")
                del self.active_connections[connection_id]
                logger.info(f"remove from active connection list: {str(connection_id)}")
                del self.active_tasks[connection_id]
                logger.info(f"remove from active tasks list: {str(connection_id)}")
        except Exception as e:
                logger.info(f"Error disconnect old connection to {connection_id}: {str(e)}")
                
            
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.active_tasks[connection_id] = set()
        logger.error(f"Client connected: {connection_id}" )
        
        
    async def disconnect(self, connection_id: str):
        """
        Disconnect a WebSocket client and clean up user sessions
        """
        # Cancel any active tasks for this connection
        await self.cancel_connection_tasks(connection_id)
        try:
                websocket = self.active_connections[connection_id]
                await websocket.close(code=1001, reason="Cleanup")
        except:
                pass
        
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
        # Clean up tasks
        if connection_id in self.active_tasks:
            del self.active_tasks[connection_id]
            
        # Clean up user sessions
        for user_id, conn_id in list(self.user_sessions.items()):
            if conn_id == connection_id:
                del self.user_sessions[user_id]
    
    def register_task(self, connection_id: str, task: asyncio.Task):
        """
        Register a task with a connection
        """
        if connection_id not in self.active_tasks:
            self.active_tasks[connection_id] = set()
        self.active_tasks[connection_id].add(task)
        
        # Set up automatic cleanup when task is done
        task.add_done_callback(
            lambda t, conn_id=connection_id: self.active_tasks[conn_id].discard(t) 
            if conn_id in self.active_tasks else None
        )
    
    async def cancel_connection_tasks(self, connection_id: str):
        """
        Cancel all tasks for a connection
        """
        cancelled_count = 0
        if connection_id in self.active_tasks:
            tasks = list(self.active_tasks[connection_id])
            cancelled_count = len(tasks)
            
            # Cancel each task
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to finish cancellation
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Clear the tasks but don't delete the entry
            self.active_tasks[connection_id] = set()
            
        return cancelled_count
            
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """
        Send a message to a specific connection
        """
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {str(e)}")
                await self.disconnect(connection_id)
                
    async def send_to_user(self,connection_id: str, user_id: str, message: Dict[str, Any]):
        """
        Send a message to a specific user via their connection
        """
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_json(message)
        else:
            logger.info(f"User {user_id} not found in sessions")
            
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all active connections
        """
        disconnected = []
        for connection_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                
                logger.error(f"Error broadcasting to {connection_id}: {str(e)}")
                disconnected.append(connection_id)
                
        # Clean up disconnected clients
        for connection_id in disconnected:
            await self.disconnect(connection_id)
            
    async def send_audio_chunk(self,connection_id: str,chunk_id: str, audio_data):
        """
        Send an audio chunk to a specific user
        """
        if connection_id in self.active_connections:
            try:
                # Convert binary data to base64 for JSON transport
                #encoded_data = base64.b64encode(audio_data)
                message = {
                    "type": "audio_chunk",
                    "chunk_id": chunk_id,
                    "data": audio_data
                }
                await self.active_connections[connection_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending audio to {connection_id}: {str(e)}")
                await self.disconnect(connection_id)
    
    async def send_chat_id(self,connection_id: str, user_id: str, chat_id: str):
        """
        Send chat ID to a specific user
        """
        if connection_id in self.active_connections:
                try:
                    message = {
                        "type": "chat_id",
                        "chat_id": chat_id
                    }
                    await self.active_connections[connection_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error sending chat_id to {user_id}: {str(e)}")
                    await self.disconnect(connection_id)

    async def is_connection_active(self, connection_id: str) -> bool:
     """
     Check if a WebSocket connection is currently active
     """
     if connection_id in self.active_connections:
        return True
     return False

    async def send_status(self,connection_id: str, status: str,text_:bool,voice:bool):
        """
        Send voice processing status updates to client
        """
        
        try:
         if text_==True:
            message = {
                "type": "text_status",
                "status": status
            }
         if voice==True:
            message = {
                "type": "voice_status",
                "status": status
            }
        
         await self.active_connections[connection_id].send_json(message)
        except Exception as e:
            logger.error(f"Error sending voice status to {connection_id}: {str(e)}")
            await self.disconnect(connection_id)

# Create a singleton instance
socket_manager = SocketManager()