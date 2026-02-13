export const runtime = "nodejs";

const ADK_API_URL = process.env.ADK_API_URL || "http://localhost:8001";
const KB_APP = "rag_agent_shard";
const KB_USER = "ui_user";
const KB_SESSION = "kb_ui_session";

export async function POST(req: Request) {
  const { query } = (await req.json()) as { query: string };

  // Ensure session exists for RAG app
  const createUrl = `${ADK_API_URL}/apps/${KB_APP}/users/${KB_USER}/sessions`;
  await fetch(createUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ id: KB_SESSION }),
    cache: "no-store",
  }).catch(() => {});

  const payload = {
    app_name: KB_APP,
    user_id: KB_USER,
    session_id: KB_SESSION,
    new_message: {
      role: "user",
      parts: [{ text: `Search the knowledge base for: ${query}. Return only the tool output and a short summary with sources.` }],
    },
    streaming: false,
  };

  const upstream = await fetch(`${ADK_API_URL}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });

  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") || "application/json", "Cache-Control": "no-store" },
  });
}
