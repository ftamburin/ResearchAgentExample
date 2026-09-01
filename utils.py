#!/usr/bin/env python
# coding: utf-8

import sys
import io
import json
import os
import urllib3
import time
import asyncio

import pdfplumber
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import Annotated, ClassVar, Sequence, TypedDict, Optional

load_dotenv()

os.environ["CORE_API_KEY"] = "WFSgPh8C1Dj6X0psRnfkqzTcVLubGeMl"



# ## Utility classes and functions
# 
# This section contains the utility classes and functions used in the workflow. It includes a wrapper around the CORE API, the Pydantic models for the input and output of the nodes, and a few general-purpose functions.
# 
# The `CoreAPIWrapper` class includes a retry mechanism to handle transient errors and make the workflow more robust.

class CoreAPIWrapper(BaseModel):
    """Simple wrapper around the CORE API."""
    base_url: ClassVar[str] = "https://api.core.ac.uk/v3"
    top_k_results: int = Field(description = "Top k results obtained by running a query on Core", default = 1)

    def _get_search_response(self, query: str) -> dict:
        api_key = os.getenv("CORE_API_KEY")
        if not api_key:
            raise RuntimeError("CORE_API_KEY is not set. Put it in the environment or a .env file.")
        http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=10, read=45))

        # Retry mechanism to handle transient errors
        max_retries = 5    
        for attempt in range(max_retries):
            response = http.request(
                'GET',
                f"{self.base_url}/search/works",
                headers={"Authorization": f"Bearer {api_key}"},
                fields={"q": query, "limit": self.top_k_results}
            )
            if 200 <= response.status < 300:
                return json.loads(response.data.decode("utf-8"))
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                detail = response.data.decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"CORE API returned HTTP {response.status}: {detail}")

    def search(self, query: str) -> str:
        response = self._get_search_response(query)
        results = response.get("results", [])
        if not results:
            return "No relevant results were found"

        # Format the results in a string
        docs = []
        for result in results:
            published_date_str = result.get('publishedDate') or result.get('yearPublished', '')
            authors_str = ' and '.join(
                item.get('name', '') if isinstance(item, dict) else str(item)
                for item in result.get('authors', [])
            )
            paper_urls = result.get('sourceFulltextUrls') or ([result.get('downloadUrl')] if result.get('downloadUrl') else [])
            docs.append((
                f"* ID: {result.get('id', '')},\n"
                f"* Title: {result.get('title', '')},\n"
                f"* Published Date: {published_date_str},\n"
                f"* Authors: {authors_str},\n"
                f"* Abstract: {result.get('abstract', '')},\n"
                f"* Paper URLs: {json.dumps(paper_urls)}"
            ))
        return "\n-----\n".join(docs)

def format_tools_description(tools: list[BaseTool]) -> str:
    return "\n\n".join([f"- {tool.name}: {tool.description}\n Input arguments: {tool.args}" for tool in tools])

async def print_stream(app: CompiledStateGraph, input: str) -> Optional[BaseMessage]:
    print("## New research running")
    print(f"### Input:\n\n{input}\n\n")
    print("### Stream:\n\n")

    # Stream the results 
    all_messages = []
    async for chunk in app.astream({"messages": [input]}, stream_mode="updates"):
        for updates in chunk.values():
            if messages := updates.get("messages"):
                all_messages.extend(messages)
                for message in messages:
                    message.pretty_print()
                    print("\n\n")
 
    # Return the last message if any
    if not all_messages:
        return None
    return all_messages[-1]

#---------------------------------------------------------------

