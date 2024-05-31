import { TodoItem } from "@/app/[username]/todo/page";
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI();
export const runtime = "edge";

export async function updateGroceryListBasedOnProductsSeenOnTheTable(
  products: string[],
  groceryList: TodoItem[]
): Promise<TodoItem[]> {
  const result = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      {
        role: "system",
        content: "The following are the products that were seen on the table:",
      },
      { role: "user", content: products.join(", ") },
      {
        role: "system",
        content: `The following is a json array of grocery list items. Return me a json array of grocery list indices of items that were seen on the table, if any. The result should be formatted like "{indices: [0, 3]}". If none of the items were seen on the table, return an empty array "{indices: []}".`,
      },
      {
        role: "user",
        content: groceryList.map((item) => item.text).join(", "),
      },
    ],
    response_format: { type: "json_object" },
  });

  const indicesObject: { indices: number[] } = JSON.parse(
    result.choices[0].message.content || "{}"
  );

  const indices = indicesObject["indices"];

  if (indices.length > 0) {
    return groceryList.map((item, index) => ({
      ...item,
      completed: indices.includes(index) || item.completed,
    }));
  }

  return groceryList;
}

interface GroceryListMatcherRequest {
  products: string[];
  groceryList: TodoItem[];
}

export async function POST(req: NextRequest) {
  const body: GroceryListMatcherRequest = await req.json();

  const { products, groceryList } = body;

  const updatedGroceryList: TodoItem[] =
    await updateGroceryListBasedOnProductsSeenOnTheTable(products, groceryList);

  return NextResponse.json({ updatedGroceryList });
}
