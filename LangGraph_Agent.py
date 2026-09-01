#!/usr/bin/env python
# coding: utf-8
"""Interactive LangGraph agent for finding, downloading, and reviewing papers.

Install dependencies:
    pip install -U langgraph langchain-core langchain-ollama pydantic \
        urllib3 pdfplumber python-dotenv

Then set CORE_API_KEY in the environment (or in .env), ensure Ollama is
running, and pull the configured model, for example: ``ollama pull qwen3.8``.

Run:
    python scientific_paper_agent_langgraph.py "find papers about ..."
"""

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
from utils import CoreAPIWrapper, format_tools_description, print_stream

load_dotenv()


class SearchPapersInput(BaseModel):
    """Input object to search papers with the CORE API."""
    query: str = Field(description="The query to search for on the selected archive.")
    max_papers: int = Field(description="The maximum number of papers to return.", default=1, ge=1, le=10)

class DecisionMakingOutput(BaseModel):
    """Output object of the decision making node."""
    requires_research: bool = Field(description="Whether the user query requires research or not.")
    answer: Optional[str] = Field(default=None, description="The answer to the user query.")

class JudgeOutput(BaseModel):
    """Output object of the judge node."""
    is_good_answer: bool = Field(description="Whether the answer is good or not.")
    feedback: Optional[str] = Field(default=None, description="Detailed feedback about why the answer is not good.")

# ## Agent state
# 
# This section defines the agent state, which contains the following information:
# - `requires_research`: Whether the user query requires research or not.
# - `num_feedback_requests`: The number of times the LLM asked for feedback.
# - `is_good_answer`: Whether the LLM's final answer is good or not.
# - `messages`: The conversation history between the user and the LLM.

class AgentState(TypedDict):
    """The state of the agent during the paper research process."""
    requires_research: bool
    num_feedback_requests: int
    is_good_answer: bool
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ## Agent tools
# 
# This section defines the tools available to the agent. The toolkit contains a tool to search for scientific papers using the CORE API, a tool to download a scientific paper from a given URL, and a tool to ask for human feedback.
# 
# To make the paper download more robust, the tool includes a retry mechanism, similar to the one used for the CORE API, as well as a mock browser header to avoid 403 errors.

@tool("search-papers", args_schema=SearchPapersInput)
def search_papers(query: str, max_papers: int = 1) -> str:
    """Search for scientific papers using the CORE API.

    Example:
    {"query": "Attention is all you need", "max_papers": 1}

    Returns:
        A list of the relevant papers found with the corresponding relevant information.
    """
    try:
        return CoreAPIWrapper(top_k_results=max_papers).search(query)
    except Exception as e:
        return f"Error performing paper search: {e}"

@tool("download-paper")
def download_paper(url: str) -> str:
    """Download a specific scientific paper from a given URL.

    Example:
    {"url": "https://sample.pdf"}

    Returns:
        The paper content.
    """
    try:        
        http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=10, read=90))
        
        # Mock browser headers to avoid 403 error
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
        }
        max_retries = 5
        for attempt in range(max_retries):
            response = http.request('GET', url, headers=headers, redirect=True)
            if 200 <= response.status < 300:
                if len(response.data) > 50 * 1024 * 1024:
                    raise ValueError("PDF exceeds the 50 MB safety limit")
                if response.data[:4] != b"%PDF":
                    content_type = response.headers.get("Content-Type", "")
                    raise ValueError(f"URL did not return a PDF (Content-Type: {content_type})")
                pdf_file = io.BytesIO(response.data)
                with pdfplumber.open(pdf_file) as pdf:
                    pages = []
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        if page_text:
                            pages.append(page_text)
                        if sum(len(item) for item in pages) >= 80_000:
                            break
                text = "\n".join(pages)[:80_000]
                if not text.strip():
                    raise ValueError("The PDF contains no extractable text (it may require OCR)")
                return text
            elif attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 2))
            else:
                detail = response.data.decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"Download returned HTTP {response.status}: {detail}")
    except Exception as e:
        return f"Error downloading paper: {e}"

tools = [search_papers, download_paper]
tools_dict = {tool.name: tool for tool in tools}


# ## Workflow nodes
# 
# This section defines the nodes of the workflow. Note how the `judge_node` is configured to end the execution if the LLM failed to provide a good answer twice to keep latency acceptable.

# LLMs (uncomment the base model you would like to use)
base_llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3.8"),
    temperature=0.0,
    num_ctx=32768
)

decision_making_llm = base_llm.with_structured_output(DecisionMakingOutput)
tool_selection_llm = base_llm.bind_tools(tools, tool_choice="required")
agent_llm = base_llm.bind_tools(tools)
judge_llm = base_llm.with_structured_output(
    JudgeOutput,
    method="json_schema",
    include_raw=True,
)

# Decision making node
def decision_making_node(state: AgentState):
    system_prompt = SystemMessage(content=(
        "You are an experienced scientific researcher.\n"
        "Your goal is to help the user with their scientific research.\n\n"
        "Based on the user query, decide if you need to perform a research "
        "or if you can answer the question directly.\n"
        "- You should perform a research if the user query requires any "
        "supporting evidence or information.\n"
        "- You should answer the question directly only for simple "
        "conversational questions, like \"how are you?\""))
    response: DecisionMakingOutput = decision_making_llm.invoke([system_prompt] + state["messages"])
    output = {"requires_research": response.requires_research}
    if not response.requires_research:
        output["messages"] = [AIMessage(content=response.answer or "How can I help with your research?")]
    return output

# Task router function
def router(state: AgentState):
    """Router directing the user query to the appropriate branch of the workflow."""
    if state["requires_research"]:
        return "planning"
    else:
        return "end"

# Planning node
def planning_node(state: AgentState):
    """Select and call the first research tool instead of returning a prose plan."""
    system_prompt = SystemMessage(content=(
        "You are the tool-selection stage of a scientific research agent.\n\n"
        f"Available tools:\n{format_tools_description(tools)}\n\n"
        "Call at least one available tool now. Do not return a prose plan or "
        "a final answer. For a topic or paper-discovery request, call "
        "search-papers. For a request about one explicit PDF URL, call "
        "download-paper using that exact URL."
    ))
    prompt = [system_prompt] + state["messages"]
    response = tool_selection_llm.invoke(prompt)

    # Some Ollama/model combinations do not honor tool_choice on the first
    # attempt. Retry once with an even narrower instruction, then fail loudly
    # instead of returning an empty update that breaks the next graph node.
    if not getattr(response, "tool_calls", None):
        retry_prompt = prompt + [HumanMessage(content=(
            "Select a tool now. Your response must contain a tool call: "
            "search-papers for research/discovery, or download-paper for one "
            "explicit PDF URL. Do not output ordinary text."
        ))]
        response = tool_selection_llm.invoke(retry_prompt)
    if not getattr(response, "tool_calls", None):
        raise RuntimeError(
            "The Ollama model did not emit a tool call. Update Ollama and "
            "langchain-ollama, and use a tool-capable model such as qwen3:8b."
        )
    return {"messages": [response]}

# Tool call node
def tools_node(state: AgentState):
    """Tool call node that executes the tools based on the plan."""
    outputs = []
    last_message = state["messages"][-1]
    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call.get("name", "")
        if tool_name not in tools_dict:
            tool_result = f"Unknown tool: {tool_name}"
        else:
            try:
                tool_result = tools_dict[tool_name].invoke(tool_call.get("args", {}))
            except Exception as exc:
                tool_result = f"Tool failed: {type(exc).__name__}: {exc}"
        outputs.append(
            ToolMessage(
                content=str(tool_result),
                name=tool_name,
                tool_call_id=tool_call["id"],
            )
        )
    if not outputs:
        raise RuntimeError("tools_node received a message without tool calls")
    return {"messages": outputs}

# Agent call node
def agent_node(state: AgentState):
    """Agent call node that uses the LLM with tools to answer the user query."""
    system_prompt = SystemMessage(content=(
        "# IDENTITY AND PURPOSE\n\n"
        "You are an experienced scientific researcher.\n"
        "Your goal is to help the user with their scientific research. "
        "You have access to a set of external tools to complete your tasks.\n"
        "Follow the plan you wrote to successfully complete the task.\n\n"
        "Add extensive inline citations to support any claim made in the answer."))
    response = agent_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Should continue function
def should_continue(state: AgentState):
    """Check if the agent should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]
    # End execution if there are no tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    else:
        return "end"

# Judge node
def judge_node(state: AgentState):
    judge_prompt = SystemMessage(content=(
        "You are an expert scientific researcher.\n"
        "Your goal is to review the final answer you provided for a "
        "specific user query.\n\n"
        "Look at the conversation history between you and the user. Based "
        "on it, you need to decide if the final answer is satisfactory or not.\n\n"
        "A good final answer should:\n"
        "- Directly answer the user query. For example, it does not answer a "
        "question about a different paper or area of research.\n"
        "- Answer extensively the request from the user.\n"
        "- Take into account any feedback given through the conversation.\n"
        "- Provide inline sources to support any claim made in the answer.\n\n"
        "In case the answer is not good enough, provide clear and concise "
        "feedback on what needs to be improved to pass the evaluation."))
    num_feedback_requests = state.get("num_feedback_requests", 0)
    if num_feedback_requests >= 2:
        return {"is_good_answer": True}
    user_query = next((message.content for message in state["messages"]
            if isinstance(message, HumanMessage)
            and not message.content.startswith("Reviewer feedback")
        ), "")
    final_answer = next((message.content for message in reversed(state["messages"])
            if isinstance(message, AIMessage)
            and not message.tool_calls), "")
    
    judge_input = [judge_prompt,
                   HumanMessage(content=(
                           f"Original user query:\n{user_query}\n\n"
                           f"Answer to evaluate:\n{final_answer}"))
                  ]
    result = judge_llm.invoke(judge_input)
    response = result["parsed"]
    if response is None:
        error = result.get("parsing_error")
        raw = result.get("raw")
        return {
            "is_good_answer": False,
            "num_feedback_requests": num_feedback_requests + 1,
            "messages": [
                HumanMessage(content=(
                        "Reviewer feedback to address: the answer could not be "
                        f"evaluated as structured JSON. Error: {error}. "
                        f"Raw response: {getattr(raw, 'content', raw)!r}"))
            ],
        }
    output = {
        "is_good_answer": response.is_good_answer,
        "num_feedback_requests": num_feedback_requests + 1,
    }
    if not response.is_good_answer:
        output["messages"] = [
            HumanMessage(content=f"Reviewer feedback to address: {response.feedback}")
        ]
    return output

# Final answer router function
def final_answer_router(state: AgentState):
    """Router to end the workflow or improve the answer."""
    if state["is_good_answer"]:
        return "end"
    else:
        return "planning"


# ## Workflow definition

# Initialize the StateGraph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("decision_making", decision_making_node)
workflow.add_node("planning", planning_node)
workflow.add_node("tools", tools_node)
workflow.add_node("agent", agent_node)
workflow.add_node("judge", judge_node)

# Set the entry point of the graph
workflow.set_entry_point("decision_making")

# Add edges between nodes
workflow.add_conditional_edges(
    "decision_making",
    router,
    {
        "planning": "planning",
        "end": END,
    }
)
workflow.add_edge("planning", "tools")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", should_continue, 
                     {"continue": "tools", "end": "judge"})
workflow.add_conditional_edges("judge",final_answer_router,
                     {"planning": "planning", "end": END})

# Compile the graph
app = workflow.compile()


# ## Example usecases for PhD academic research
#
# "Download and summarise the findings of this paper: https://ceur-ws.org/Vol-4112/104_main_long.pdf"
# "Search 4 papers on quantum machine learning?"
#

async def main(test_input):
    # Run tests and store the results
    final_answer = await print_stream(app, test_input)
    if final_answer is None:
        raise RuntimeError("The graph completed without producing a message")
    output = final_answer.content
    # Display results
    print(f"## Input:\n\n{test_input}\n\n")
    print(f"## Output:\n\n{output}\n\n")

if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:])))
