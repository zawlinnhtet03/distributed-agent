import { NextRequest, NextResponse } from "next/server";

const ADK_API_URL = process.env.ADK_API_URL || "http://localhost:8001";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    
    const res = await fetch(`${ADK_API_URL}/kb/upload`, {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) {
      const errorText = await res.text();
      return NextResponse.json(
        { success: false, error: `Backend error: ${errorText}` },
        { status: res.status }
      );
    }
    
    const data = await res.json();
    return NextResponse.json(data);
    
  } catch (error: any) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { success: false, error: error?.message || "Upload failed" },
      { status: 500 }
    );
  }
}
