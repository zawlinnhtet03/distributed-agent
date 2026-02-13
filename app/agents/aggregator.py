"""
Aggregator Agent - Coordinator and Result Synthesizer

Step 2 in the pipeline. Reads the plan from the Planner (state["plan"]),
delegates to the appropriate agents, and synthesizes the final response.
"""

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent

from app.agents.guardian import guardian_agent, guardian_pre_agent
from app.agents.planner import planner_agent
from app.agents.rag import rag_agent
from app.agents.scraper import scraper_agent
from app.agents.data import data_agent
from app.agents.video import video_agent
from app.model_factory import ModelFactory


# ============================================================================
# LAYER: GatheringLayer (ParallelAgent)
# Runs web search + knowledge retrieval SIMULTANEOUSLY
# ============================================================================
gathering_layer = ParallelAgent(
    name="gathering_layer",
    description="Research layer — runs scraper and rag in parallel for web search and knowledge retrieval",
    sub_agents=[scraper_agent, rag_agent],
)


AGGREGATOR_INSTRUCTION = """You are the Aggregator Agent — the coordinator and synthesizer.

The Planner has already analyzed the user's request and produced an execution plan.
You can read the plan from: {plan}

The user's request may have been sanitized (PII redacted) by the Pre-Guardian.
If present, use this sanitized request as the source of truth: {sanitized_request}

Your Job:
1. READ the plan from the Planner (above)
2. DELEGATE tasks to the right agents according to the plan
3. SYNTHESIZE all agent results into a single, cohesive response

Hard Requirements:
- You MUST execute EVERY step in the Planner plan unless it is impossible.
- If a step says agent="gathering_layer" you MUST delegate to **gathering_layer**.
- If a step says agent="video" you MUST delegate to **video**.
- For PARALLEL steps: delegate to all required agents first, then wait for their outputs before writing your final response.
- If a delegated agent returns no useful output or an error, explicitly state that in your response (do not silently skip the step).

Available Agents (use EXACT names when delegating):
- **gathering_layer**: Research (web search + knowledge retrieval in parallel)
- **data**: Data analysis, EDA, cleaning, modeling
- **video**: Local video file analysis

Execution Rules:
- If the plan says agent="aggregator", respond directly yourself (simple queries)
- If the plan assigns gathering_layer, delegate to gathering_layer
- If the plan assigns data, delegate to data
- If the plan assigns video, delegate to video
- If the plan has PARALLEL steps, delegate to those agents (they run independently)
- If the plan has SEQUENTIAL steps, follow the dependency order

Synthesis Rules:
- Combine complementary information into a cohesive response
- If sources conflict, mention the disagreement
- Preserve source URLs when provided
- Keep the response concise and well-structured
- Include relevant metrics/stats from data analysis
- Reference video analysis findings when applicable
"""


aggregator_agent = LlmAgent(
    name="aggregator",
    model=ModelFactory.create(tier="default", temperature=0.5),
    instruction=AGGREGATOR_INSTRUCTION,
    description="Reads the planner's plan, delegates to agents, and synthesizes the final response",
    sub_agents=[gathering_layer, video_agent, data_agent],
)


# ============================================================================
# ROOT: DecisionPipeline (SequentialAgent)
# Step 1: Planner → Step 2: Guardian Pre → Step 3: Aggregator (delegates & synthesizes) → Step 4: Guardian
# ============================================================================
decision_pipeline = SequentialAgent(
    name="decision_pipeline",
    description="Sequential pipeline: planner → guardian_pre → aggregator → guardian",
    sub_agents=[
        planner_agent,      # Step 1: Analyze query and produce a plan
        guardian_pre_agent, # Step 2: Sanitize user request (PII redaction) before delegation
        aggregator_agent,   # Step 3: Execute plan and synthesize results
        guardian_agent,     # Step 4: Safety review
    ],
)

# Export for ADK
root_agent = decision_pipeline
