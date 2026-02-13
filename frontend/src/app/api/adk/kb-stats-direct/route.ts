export const runtime = "nodejs";

const RAG_PORT = 8003;

export async function GET() {
  try {
    // Call a simple stats endpoint on the RAG agent
    const res = await fetch(`http://localhost:${RAG_PORT}/stats`, {
      method: "GET",
      cache: "no-store",
    });

    if (res.ok) {
      const data = await res.json();
      return Response.json(data);
    }

    // Fallback: try to get stats from the agent's health check or similar
    return Response.json({
      totalDocs: 0,
      collection: "research_knowledge_base",
      model: "all-MiniLM-L6-v2",
      byType: {},
      error: "Stats endpoint not available"
    });
  } catch (e: any) {
    return Response.json({
      totalDocs: 0,
      collection: "research_knowledge_base", 
      model: "all-MiniLM-L6-v2",
      byType: {},
      error: String(e?.message ?? e)
    });
  }
}
