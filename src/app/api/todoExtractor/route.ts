import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI();
export const runtime = "edge";

export async function extractTodoFrom(text: string): Promise<string> {
  const result = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      {
        role: "system",
        content:
          "The following is a voice command recorded form a voice assistant, extract the todo item from it to insert it in a todo list. For example, if the voice command is 'Add 'Buy milk' to my shopping list', the extracted todo item is 'Buy milk'.",
      },
      { role: "user", content: text },
    ],
  });

  return result.choices[0].message.content || "";
}

interface TodoExtractorRequest {
  text: string;
}

export async function POST(req: NextRequest) {
  const { text }: TodoExtractorRequest = await req.json();

  const extractedTodo = await extractTodoFrom(text);

  return NextResponse.json({ text: extractedTodo });
}
