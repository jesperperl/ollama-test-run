from openai import OpenAI
import json

client = OpenAI(
    base_url='http://127.0.0.1:11434/v1',
    api_key='1234',
)


# Example function to query ChatGPT
def ask_chatgpt(messages):
    response = client.chat.completions.create(
        model="gemma3:4b",
        messages=messages,
        temperature=0.7,
        )

    response_model = response.model_dump()
    print(json.dumps(response_model, indent=4))

    return response.choices[0].message.content


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
        },
    {
        "role": "user",
        "content": "What is the captial of France?"
        },
    {
        "role": "assistant",
        "content": "The capital of France is Paris."
        },
    {
        "role": "user",
        "content": "What is an interesting fact of Paris."
        }
    ]
response = ask_chatgpt(messages)
print(response)
