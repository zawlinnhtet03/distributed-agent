import os
from tavily import TavilyClient
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types
from dotenv import load_dotenv
import litellm
from google.adk.models.lite_llm import LiteLlm

litellm.set_verbose = False
litellm.suppress_debug_info = True

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("Missing MISTRAL_API_KEY")

llm_model = LiteLlm(
    model="mistral/mistral-medium-latest",
    api_key=mistral_api_key,
)

tavily_api_key = os.getenv("TAVILY_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not tavily_api_key:
    raise ValueError("Missing TAVILY_API_KEY")

tavily_client = TavilyClient(api_key=tavily_api_key)
retry_config = types.HttpRetryOptions(attempts=5, initial_delay=1, http_status_codes=[429, 500, 503, 504])

def analyze_market_trends(query: str) -> str:
    """Performs market research using Tavily."""
    try:
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )
        sources = "\n".join([f"- {r['title']}: {r['url']}" for r in response['results']])
        return f"SUMMARY: {response['answer']}\n\nDETAILS: {str(response['results'])[:1000]}\n\nSOURCES:\n{sources}"
    except Exception as e:
        return f"Error: {e}"

# 4. Define the Agent
scraping_agent = LlmAgent(
    model=llm_model,
    name="scraping_agent_shard",
    instruction="""
    You are the Scraping Agent.
    ALWAYS call 'analyze_market_trends' to find live data.


    Rules:
    1) Make 1-3 targeted queries.
    2) After the tool returns, produce ONE final structured Markdown report.
    3) Do not output partial tool-call text, placeholders, or phrases like "pending".
    4) ONLY use information from the tool output. Do NOT infer from any video content or any other agent output.
    5) Do not claim "AI"/"machine learning" unless the tool output explicitly supports it.

    Output format (Markdown):
    - Trend Summary (3-6 bullets)
    - Key Evidence (2-5 bullets with concrete facts)
    - Sources (bulleted list of URLs strictly copied from the tool output; if none are present, write "Sources: None")
    """,
    tools=[analyze_market_trends]
)

# 5. EXPOSE THE APP (Crucial Step!)
# This variable 'app' is what uvicorn looks for
app = to_a2a(scraping_agent, port=8001)

if __name__ == "__main__":
    # Optional: Allow running with `python scraping_server.py` for testing
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)