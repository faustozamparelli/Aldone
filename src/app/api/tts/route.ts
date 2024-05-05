import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI();

export async function POST(req: NextRequest) {
  const { text } = await req.json();

  const mp3 = await openai.audio.speech.create({
    model: "tts-1-hd",
    voice: "shimmer",
    input: text,
  });

  return new NextResponse(await mp3.arrayBuffer(), {
    headers: { "Content-Type": "audio/mp3" },
  });
}
