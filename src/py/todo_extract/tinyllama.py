import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device=0,  # Use GPU if available
)

while True:
    # Get the user message from console input
    user_message = input("You: ")

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "Extract the todo from the input summarizing it keep the verb and the object and location. Respond with just that no more words",
        },
        {"role": "user", "content": user_message},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(prompt, max_new_tokens=4)  # Limit the length of the generated text

    # Extract only the assistant's response from the generated text
    assistant_response = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
    print("Assistant: ", assistant_response)
