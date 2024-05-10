# Import Modules
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai = OpenAI()

# Build completion objects
completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello ChatGPT!"},
    ],
)

# Print the response to screen
print(completion.choices[0].message)
