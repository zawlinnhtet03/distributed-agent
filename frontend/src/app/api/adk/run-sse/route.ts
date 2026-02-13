export const runtime = "nodejs";

const ADK_API_URL = process.env.ADK_API_URL || "http://localhost:8001";

export async function POST(req: Request) {
  const bodyJson = await req.json();
  
  // Transform camelCase to snake_case for ADK API
  const adkPayload = {
    app_name: bodyJson.appName,
    user_id: bodyJson.userId,
    session_id: bodyJson.sessionId,
    new_message: bodyJson.newMessage,
    streaming: bodyJson.streaming ?? true,
  };

  let upstream: Response;
  try {
    upstream = await fetch(`${ADK_API_URL}/run_sse`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(adkPayload),
      cache: "no-store",
    });
  } catch (err: any) {
    return new Response(
      JSON.stringify({
        error: "Failed to reach agent backend",
        adkApiUrl: ADK_API_URL,
        detail: String(err?.message ?? err),
      }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || "text/event-stream",
      "Cache-Control": "no-store",
      Connection: "keep-alive",
    },
  });
}
