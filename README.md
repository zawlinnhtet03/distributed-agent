# distributed-agent

Step-by-step setup and run guide for the distributed market/video analysis agents.

## 1) Prerequisites
- Python 3.10+ and Git installed
- Ollama installed and running (Windows: install from https://ollama.com/download; start the Ollama app or run `ollama serve`)
- Pull the local vision model: `ollama pull moondream`
- Groq API key (required), Tavily API key (required), Google API key (optional for future Google LLM usage)

## 2) Clone and install dependencies
```bash
git clone <repo-url>
cd distributed-agent
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## 3) Environment variables
Create a `.env` file in the project root:
```bash
GROQ_API_KEY="your_groq_key"          # required (LiteLlm via Groq)
TAVILY_API_KEY="your_tavily_key"      # required (web search)
GOOGLE_API_KEY="your_google_key"      # optional (if you switch to Gemini)
```

## 4) Start the agent shards
Open two terminals (activate the venv in each):
- Scraping shard (ports to 8001):
	```bash
	python scraper.py
	```
- Video shard (ports to 8002):
	```bash
	python video.py
	```

## 5) Run the orchestrator UI (ADK Web)
From the project root in a new terminal (venv activated):
```bash
adk web .
```
This loads the ADK web console, which connects to the two running shards via their A2A endpoints.

## 6) Running the scripts directly
- To re-run only the web scraping flow (without the UI), keep `scraper.py` running and send requests to `http://localhost:8001/.well-known/agent-card.json` via your own orchestrator or tests.
- To re-run only the video analysis locally, run `python video.py` and ensure `ollama serve` is running with the `moondream` model pulled.

## 7) Notes and tips
- The LiteLlm model is set to `openai/llama-3.1-8b-instant` for scraping and `openai/meta-llama/llama-4-scout-17b-16e-instruct` for video summarization. You can switch models in `scraper.py` and `video.py` if needed.
- If OpenCV fails to open a video, check the codec or try re-encoding the file.
- If Tavily limits are hit, lower `max_results` or reduce queries inside `scraper.py`.