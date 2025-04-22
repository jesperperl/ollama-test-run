from openai import OpenAI

client = OpenAI(
    base_url='http://127.0.0.1:11434/v1',
    api_key='1234',
)

def ask_gpt(user_message: str):
    response = client.chat.completions.create(
        model="gemma3:4b",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': "What is the capital of Denmark?"},
            {'role': 'assistant', 'content': "The capital of Denmark is Copenhagen"},
            {'role': 'user', 'content': user_message},
        ],
        temperature=0.3,
    )
    print(response)
    return response.choices[0].message.content

user = "What is an interesting fact about Copenhagen?"
response = ask_gpt(user)
print(response)
