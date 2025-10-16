import os
import asyncio
import aiohttp
import base64
import ssl
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from ..utils import config  # Import the config module
import re  # Added for re.split
# Force SSL bypass for VPN/V2ray compatibility with TTS API
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['SSL_VERIFY'] = 'false'
# Configure SSL to be more permissive for proxy environments
ssl._create_default_https_context = ssl._create_unverified_context
import time
from app.utils.socket_manager import socket_manager  # Import the socket manager



class VoiceService:
    """Service for generating voice audio files from text using a TTS API with WebSocket support."""
    
    _cancellation_flags: Dict[str, bool] = {}

    def __init__(self, 
                 language: str = "en-US", 
                 model: str = "simba-base",
                 base_url: str = None,
                 ):
        self.url = config.TTS_URL
        self.headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "Authorization": config.TTS_KEY
        }
        self.voice_id = "emily"
        self.language = language
        self.audio_format = "mp3"
        self.model = model
        
      
    async def process_audio_file(self, id: str, chunk_id:str,message_input: str, language, start_time,end:bool,connection_id: str = None):
        """Background task to process and save the audio file with WebSocket streaming"""
        try:
            
            if language == 'ar_SA':
                self.language = "ar-AE"
                self.model = "simba-multilingual"
                
            payload = {
                "audio_format": self.audio_format,
                "input": message_input,
                "language": self.language,
                "model": self.model,
                "options": {"loudness_normalization": True, "speed": 1,
                            "volume": 1.0},
                "voice_id": self.voice_id
            }
            
            # Create connector with SSL disabled for VPN/V2ray compatibility
            connector = aiohttp.TCPConnector(ssl=False)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(self.url, json=payload, headers=self.headers) as response:
                    response.raise_for_status()
                    json_data = await response.json()

            if "audio_data" not in json_data:
                print(f"Error: 'audio_data' not found in the API response.")
                if self.user_id:
                    await socket_manager.send_status(connection_id=connection_id, status="error:"+chunk_id,text_=False,voice=True)
                return False
                    
            print(f"+")    
            audio_data_encoded = json_data["audio_data"]
            print(f"chunk Id created: {chunk_id},user :{connection_id} ")
            
            await socket_manager.send_audio_chunk(connection_id=connection_id,chunk_id=chunk_id, audio_data=audio_data_encoded)
           
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"chuk: {elapsed_time}")
                #await socket_manager.send_voice_status(self.user_id, "completed", id)
                
        except Exception as e:
            print(f"Error in background audio processing: {str(e)}")
            if connection_id:
                await socket_manager.send_status(connection_id=connection_id, status="error create voice for user"+id,text_=False,voice=True)

        return None
    
    @staticmethod
    def should_cancel_response(connection_id: str) -> bool:
        return VoiceService._cancellation_flags.get(connection_id, False)

    @staticmethod
    def set_cancel_flag(connection_id: str, cancel: bool):
        VoiceService._cancellation_flags[connection_id] = cancel

    async def process_response_parts_for_voice(self, complete_response: str, language: str,connection_id: str = None):
        parts = re.split(r'[.?؟]', complete_response)
        voice_tasks = []
        counter = 0

        for part in parts:
            if not part.strip():  # Skip empty parts
                continue

            # Check for cancellation before voice processing
            if VoiceService.should_cancel_response(connection_id=connection_id):
                raise asyncio.CancelledError("Response cancelled")
                                        
            start_time = time.time()
            if counter == 0:
                if part.strip(): # Ensure part is not empty before sending
                    await self.process_audio_file(f"{connection_id}", str(counter), part.strip(), language, start_time, False,connection_id=connection_id)
            else:
                if part.strip(): # Ensure part is not empty before creating task
                    task = asyncio.create_task(self.process_audio_file(f"{connection_id}", str(counter), part.strip(), language, start_time, False,connection_id=connection_id))
                    voice_tasks.append(task)
            counter = counter + 1
        
        tasks=[]
        tasks=voice_tasks
    
        if tasks and 1<counter:
                pending_tasks = [t for t in tasks if not t.done()]
                if pending_tasks:
                    print(f"Waiting for {len(pending_tasks)} remaining tasks...")
                    await asyncio.gather(*pending_tasks)
                    print("All remaining  tasks completed")
                    
        return True