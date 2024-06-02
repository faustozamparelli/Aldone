import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";

export async function POST(req: NextRequest) {
  const file = await fs.readFile(process.cwd() + "/.seen");
  return NextResponse.json({ foods: file.toString().split("\n") });
}
