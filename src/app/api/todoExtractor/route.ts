import { TodoItem } from "@/app/[username]/page";
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI();
export const runtime = "edge";

export interface TodoExtractorRequest {
  text: string;
  groceryList: TodoItem[];
  todoList: TodoItem[];
}

export interface TodoExtractorResponse {
  addingTodo?: { id: string; text: string };
  completingTodo?: { category: "todo" | "grocery"; id: string };
}

async function extractTodoFrom(
  text: string,
  todoList: TodoItem[],
  groceryList: TodoItem[]
): Promise<TodoExtractorResponse> {
  const result = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      { role: "system", content: "The following is the user's todo list:" },
      { role: "user", content: JSON.stringify(todoList) },

      { role: "system", content: "The following is the user's grocery list:" },
      { role: "user", content: JSON.stringify(groceryList) },
      {
        role: "system",
        content: `The following is a voice command recorded form a voice assistant.

        If the user is asking to mark a to-do as complete or to cross something out of the grocery list, return a json object formatted exactly like {completingTodo: {category: "todo"|"grocery", id: string}} where category represents the list where the item belongs, either "todo" or "grocery", and id is the id of the existing item to mark as completed within that list. For example, {completingTodo: {category: "todo", id: "this_beautiful_task"}}.

        Otherwise, the user is asking to add a new item. The text is the extracted item and the id is the "nice_readable_id" for it. For example, if the voice command is 'Add 'Buy milk' to my shopping list', the extracted todo item is 'Milk', or if the voice command is "Remind me to do the laundry", the extracted todo item is "Do the laundry". You should also generate a "nice_readable_id" and finally return a json object structured exacly as {addingTodo: {text: string; id: string}}. For example, {addingTodo: {text: 'Do the laundry', id: 'laundry_do'}}.`,
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

function capitalizeFirstLetter(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export async function POST(req: NextRequest) {
  const { text, groceryList, todoList }: TodoExtractorRequest =
    await req.json();
  const todoExtractorResponse: TodoExtractorResponse = await extractTodoFrom(
    text,
    todoList,
    groceryList
  );
  const result: TodoExtractorResponse = todoExtractorResponse.addingTodo
    ? {
        addingTodo: {
          id: todoExtractorResponse.addingTodo.id,
          text: capitalizeFirstLetter(todoExtractorResponse.addingTodo.text),
        },
      }
    : {
        completingTodo: todoExtractorResponse.completingTodo,
      };
  console.log({
    todoExtractorRequest: { text },
    todoExtractorResponse: result,
  });
  return NextResponse.json(result);
}
