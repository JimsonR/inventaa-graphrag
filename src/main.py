#!/usr/bin/env python3
"""
FastAPI Service for Hybrid RAG Agent.
Main application entry point.
"""

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.endpoints.retriever_api import router as retriever_router
from src.endpoints.router_api import router as message_router
from src.services.retrieve import initialize_agent

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at application startup and once at shutdown.
    Pre-loads the LangChain Agent, LLMs, and Graph stores so the first
    incoming request is never delayed by a cold load.
    """
    logger.info("Starting Hybrid RAG Agent initialization...")
    initialize_agent()
    logger.info("Hybrid RAG Agent is ready.")
    yield


app = FastAPI(
    title="Hybrid RAG Agent API",
    description="FastAPI service serving a LangChain Azure OpenAI Agent connected to a Neo4j Graph.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins so Swagger UI and external clients can reach the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root_redirect():
    """
    Redirects root requests directly to the interactive Swagger documentation.
    """
    return RedirectResponse(url="/docs")

# Include API endpoints
app.include_router(retriever_router)
app.include_router(message_router)

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8080, reload=True)

