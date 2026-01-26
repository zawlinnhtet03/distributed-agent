import os

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.models.lite_llm import LiteLlm


llm_model = LiteLlm(
    model="openai/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    api_base="https://api.groq.com/openai/v1",
)


aggregator_agent = LlmAgent(
    model=llm_model,
    name="AggregatorAgent",
    instruction=r"""
    You are the Lead Market Analyst.

    You will receive outputs from two upstream agents:
    - scraping_agent_shard: web trend report
    - video_agent_shard: visual/video analysis

    Hard rules:
    - Do NOT invent sources.
    - If the scraping report has \"Sources: None\" or indicates an error/limited data, you MUST lower confidence and you MAY NOT make strong claims.
    - Do not claim \"AI\"/\"machine learning\" unless the scraping report explicitly supports it.

    Task:
    1) Summarize each input separately.
    2) Correlate: do visuals support the web trend report?
    3) Produce a verdict tag from EXACTLY one of:
       - Real & Observable
       - Hype vs Reality Mismatch
       - Insufficient Evidence
    4) Provide Confidence: High / Medium / Low.

    Output format (Markdown):
    - Scraping Agent Summary
    - Video Agent Summary
    - Correlation (include 2-5 bullets of evidence; only cite evidence present in inputs)
    - Verdict Tag
    - Confidence
    - Executive Summary (120-180 words)
    - Final Verdict (one line)
    """,
)


remote_scraping_agent = RemoteA2aAgent(
    name="scraping_agent_shard",
    agent_card="http://localhost:8001/.well-known/agent-card.json",
)

remote_video_agent = RemoteA2aAgent(
    name="video_agent_shard",
    agent_card="http://localhost:8002/.well-known/agent-card.json",
)


gathering_squad = ParallelAgent(
    name="GatheringLayer",
    sub_agents=[remote_scraping_agent, remote_video_agent],
)


root_agent = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[gathering_squad, aggregator_agent],
)
