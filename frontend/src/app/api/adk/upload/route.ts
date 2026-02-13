export const runtime = "nodejs";

const ADK_API_URL = process.env.ADK_API_URL || "http://localhost:8001";

export async function POST(req: Request) {
  const formData = await req.formData();

  let upstream: Response;
  try {
    upstream = await fetch(`${ADK_API_URL}/upload`, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });
  } catch (err: any) {
    return new Response(
      JSON.stringify({
        success: false,
        error: "Failed to reach agent backend",
        adkApiUrl: ADK_API_URL,
        detail: String(err?.message ?? err),
      }),
      { status: 502, headers: { "Content-Type": "application/json" } },
    );
  }

  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") || "application/json",
      "Cache-Control": "no-store",
    },
  });
}
