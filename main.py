from fastapi.middleware.cors import CORSMiddleware 
from fastapi import FastAPI
import uvicorn
import logging
from app.api.router_factory import RouterFactory
from app.services.embedding_service import has_new_documents
from scripts.index_documents import run as run_indexing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medicall Chatbot API",
    description="An API for interacting with a Medical chatbot.",
    version="1.0",  
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    
)


main_router = RouterFactory.create_main_router()
app.include_router(main_router)

# Conditionally run document indexing
if has_new_documents():
    logging.info("New documents detected or index is empty. Running indexing...")
    run_indexing()
else:
    logging.info("No new documents detected. Index is up to date.")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)