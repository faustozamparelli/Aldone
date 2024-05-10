import { NextRequest, NextResponse } from "next/server";

export interface ProcessVoiceRequest {
  input: string[];
}

export interface ProcessVoiceResponse {
  action: any;
  agentReply: string;
}

export default async function POST(req: NextRequest) {
  const { input }: ProcessVoiceRequest = await req.json();

  const result: ProcessVoiceResponse = {
    action: "todo",
    agentReply: "Ok, done.",
  };

  return NextResponse.json(result);
}
