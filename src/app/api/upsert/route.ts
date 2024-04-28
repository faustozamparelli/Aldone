import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  //   const body = await req.json();

  return NextResponse.json({ success: true });
}
