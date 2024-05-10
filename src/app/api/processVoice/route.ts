import { NextRequest, NextResponse } from "next/server";

export interface ProcessVoiceRequest {
  input: string[];
}

interface VoiceAction {
  addingTodo?: string;
  completingTodoIndex?: number;
  removingTodoIndex?: number;
}

export interface ProcessVoiceResponse {
  action: VoiceAction;
  agentReply: string;
}

export default async function POST(req: NextRequest) {
  const { input }: ProcessVoiceRequest = await req.json();

  const result: ProcessVoiceResponse = {
    action: { addingTodo: "To-do processed correctly" },
    agentReply: "Ok, done.",
  };

  return NextResponse.json(result);
}
