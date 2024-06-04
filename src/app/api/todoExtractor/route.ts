import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI();
export const runtime = "edge";

async function extractTodoFrom(text: string): Promise<TodoExtractorResponse> {
  const result = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      {
        role: "system",
        content: `The following is a voice command recorded form a voice assistant, extract the todo item from it to insert it in a todo list. For example, if the voice command is 'Add 'Buy milk' to my shopping list', the extracted todo item is 'Buy milk'. You should also generate a "nice_readable_id" and finally return a json object structured exacly as {text: string; id: string}. For example, {text: 'Buy milk', id: 'milk_buy'}.`,
      },
      { role: "user", content: text },
    ],
    response_format: { type: "json_object" },
  });

  const resultObject: TodoExtractorResponse = JSON.parse(
    result.choices[0].message.content || "{}"
  );

  return resultObject;
}

export interface TodoExtractorRequest {
  text: string;
}

export interface TodoExtractorResponse {
  text: string;
  id: string;
}

function capitalizeFirstLetter(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export async function POST(req: NextRequest) {
  const { text }: TodoExtractorRequest = await req.json();
  const todoExtractorResponse: TodoExtractorResponse = await extractTodoFrom(
    text
  );
  const result: TodoExtractorResponse = {
    text: capitalizeFirstLetter(todoExtractorResponse.text),
    id: todoExtractorResponse.id,
  };
  console.log({
    todoExtractorRequest: { text },
    todoExtractorResponse: result,
  });
  return NextResponse.json(result);
}
