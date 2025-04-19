from openai import AsyncOpenAI
# import chainlit as cl

# import openai

# client = AsyncOpenAI(
#     base_url='http://127.0.0.1:11434/v1',
#     api_key='1234',
# )

# openai.api_key = '1234'
# openai.base_url = 'http://127.0.0.1:11434/v1'

# user_message = 'What is the weather in Seattle today?'

# response = openai.chat.completions.create(
#     # model='gpt-4',
#     model='gemma3:4b',
#     messages=[
#         {'role': 'system', 'content': 'You are a helpful assistant.'},
#         {'role': 'user', 'content': user_message},
#     ]
# )
# print(response.choices[0].message.content)

from agents import Agent, Runner, OpenAIChatCompletionsModel

model = OpenAIChatCompletionsModel(
    model="gemma3:4b",
    openai_client=AsyncOpenAI(
        base_url='http://127.0.0.1:11434/v1',
        api_key='1234',
    )
)

agent = Agent(
    name='Assistant',
    instructions='You are a helpful assistant',
    model=model,
)

result = Runner.run_sync(agent, "Create a meal plan for a week.")

print(result.final_output)
