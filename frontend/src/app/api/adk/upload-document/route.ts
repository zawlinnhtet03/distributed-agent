import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await req.formData();
    
    // Forward to RAG agent
    const res = await fetch("http://localhost:8003/upload", {
      method: "POST",
      body: formData,
    });
    
    if (!res.ok) {
      const errorText = await res.text();
      return NextResponse.json(
        { success: false, error: `RAG agent error: ${errorText}` },
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

export const config = {
  api: {
    bodyParser: false, // Required for handling file uploads
  },
};
