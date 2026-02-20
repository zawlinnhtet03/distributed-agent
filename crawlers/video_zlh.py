import os
import cv2
import ollama
from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types
import litellm
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent

litellm.set_verbose = False
litellm.suppress_debug_info = True

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY")

llm_model = LiteLlm(
    # model="openai/llama-3.1-8b-instant",
    model="openai/meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
    api_base="https://api.groq.com/openai/v1" 
)

def analyze_video_locally(video_path: str) -> str:
    """
    Analyzes video with explicit error checking.
    Uses local OpenCV + Ollama (moondream) for inference.
    """
    print(f"🎥 Video Shard: Processing '{video_path}'...")
    
    # Path Check
    if not os.path.exists(video_path):
        return f" Error: Video file not found at {video_path}"

    try:
        # Open Video
        video = cv2.VideoCapture(video_path)
        
        # Check if OpenCV opened it successfully
        if not video.isOpened():
            return " Error: OpenCV could not open the file. It might be corrupt or the codec is missing."

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # Prevent division by zero
        if fps <= 0:
            return " Error: Could not determine video FPS. Video metadata might be missing."

        duration = total_frames / fps
        print(f"⏳ Video loaded! Duration: {duration:.1f}s")
        
        # Extract Frames (20%, 50%, 80%)
        indices = [int(total_frames * 0.2), int(total_frames * 0.5), int(total_frames * 0.8)]
        descriptions = []
        
        for i, frame_idx in enumerate(indices):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if not success: 
                print(f"⚠️ Failed to read frame {i+1}")
                continue

            # Encode for Ollama
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            print(f"🦙 moondream is looking at Frame {i+1}...")
            response = ollama.chat(
                model='moondream',
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in detail.',
                    'images': [image_bytes]
                }]
            )
            desc = response['message']['content']
            descriptions.append(f"**Timestamp {frame_idx/fps:.1f}s:** {desc}")
            
        video.release()
        
        if not descriptions:
            return " Error: Video opened, but no frames could be read."

        return (
            f"✅ Local Video Analysis:\n"
            f"⏱️ **Duration:** {duration:.1f}s\n"
            f"🎞️ **Visual Narrative:**\n" + "\n\n".join(descriptions)
        )

    except Exception as e:
        return f" Python Error: {e}"

# 4. Define the Agent
video_agent_shard = LlmAgent(
    model=llm_model,
    name="video_agent_shard", # Matches the name used in your Orchestrator
    instruction="""
    You are the Video Forensics Engineer.
    Your tool runs LOCALLY on the machine.
    
    Task:
    1. Run `analyze_video_locally`.
    2. Read the text descriptions returned by the tool.
    3. Synthesize the findings into a "Vibe Check" report.

    Rules:
    - Always call `analyze_video_locally` exactly once.
    - After the tool returns, produce ONE final Markdown report.
    - Do not output partial tool-call text, placeholders, or phrases like "pending".

    Output format (Markdown):
    - Video Summary (2-5 bullets)
    - Key Visual Evidence (bullets)
    - Notable Text/Logos Seen (if any)
    """,
    tools=[analyze_video_locally]
)

# 5. Expose via A2A
# This 'app' variable is what Uvicorn runs
app = to_a2a(video_agent_shard, port=8002)

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to ensure it listens on all interfaces if running inside Docker/VM
    uvicorn.run(app, host="localhost", port=8002)