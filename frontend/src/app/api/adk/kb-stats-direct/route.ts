export const runtime = "nodejs";

const ADK_API_URL = process.env.ADK_API_URL || "http://localhost:8001";

export async function GET() {
  try {
    const res = await fetch(`${ADK_API_URL}/kb/stats`, {
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
      collection: "router_multimodal_items",
      model: "hash-v1",
      byType: {},
      error: `Stats endpoint not available at ${ADK_API_URL}/kb/stats`
    });
  } catch (e: any) {
    return Response.json({
      totalDocs: 0,
      collection: "router_multimodal_items", 
      model: "hash-v1",
      byType: {},
      error: String(e?.message ?? e)
    });
  }
}
