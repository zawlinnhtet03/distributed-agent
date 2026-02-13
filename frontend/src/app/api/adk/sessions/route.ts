export const runtime = "nodejs";

const ADK_API_URL = process.env.ADK_API_URL || "http://localhost:8001";

// Create a session
export async function POST(req: Request) {
  const body = await req.json();
  const { appName, userId, sessionId } = body as {
    appName: string;
    userId: string;
    sessionId?: string;
  };

  if (!appName || !userId) {
    return new Response(JSON.stringify({ error: "appName and userId are required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Create session using ADK API pattern
  const createUrl = `${ADK_API_URL}/apps/${appName}/users/${userId}/sessions`;
  
  try {
    const upstream = await fetch(createUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: sessionId ? JSON.stringify({ id: sessionId }) : "{}",
      cache: "no-store",
    });

    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "application/json",
        "Cache-Control": "no-store",
      },
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
}

// Get session info
export async function GET(req: Request) {
  const url = new URL(req.url);
  const appName = url.searchParams.get("appName");
  const userId = url.searchParams.get("userId");
  const sessionId = url.searchParams.get("sessionId");

  if (!appName || !userId || !sessionId) {
    return new Response(JSON.stringify({ error: "appName, userId, and sessionId are required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const getUrl = `${ADK_API_URL}/apps/${appName}/users/${userId}/sessions/${sessionId}`;
  
  try {
    const upstream = await fetch(getUrl, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
      cache: "no-store",
    });

    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") || "application/json",
        "Cache-Control": "no-store",
      },
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
}
