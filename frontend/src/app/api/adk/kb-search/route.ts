export const runtime = "nodejs";

const ADK_API_URL = process.env.ADK_API_URL || "http://localhost:8001";

export async function POST(req: Request) {
  const { query, nResults, top_k } = (await req.json()) as {
    query: string;
    nResults?: number;
    top_k?: number;
  };
  const topK = Number.isFinite(top_k) ? Number(top_k) : Number.isFinite(nResults) ? Number(nResults) : 5;

  const upstream = await fetch(`${ADK_API_URL}/kb/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
    cache: "no-store",
  });

  const data = await upstream.json().catch(() => ({}));
  const docs = Array.isArray(data?.documents) ? data.documents : [];

  const formatted = docs
    .map((doc: any, idx: number) => {
      const type = String(doc?.type || "note").toUpperCase();
      const source = String(doc?.source || "unknown");
      const content = String(doc?.content || "");
      const score = Math.round((Number(doc?.score || 0) || 0) * 100);
      return `**[${idx + 1}] ${type} - ${source}** (Relevance: ${score}%)\n${content}`;
    })
    .join("\n\n");

  const normalized = {
    query: String(query || ""),
    results: docs.map((doc: any) => ({
      content: String(doc?.content || ""),
      source: String(doc?.source || "unknown"),
      type: String(doc?.type || "note").toLowerCase(),
      score: Number(doc?.score || 0) || 0,
      doc_id: String(doc?.doc_id || ""),
    })),
    total_found: Number(data?.total_found || docs.length || 0),
    response: formatted,
    warnings: Array.isArray(data?.warnings) ? data.warnings : [],
    error: data?.error ? String(data.error) : undefined,
  };

  return new Response(JSON.stringify(normalized), {
    status: upstream.ok ? 200 : upstream.status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
