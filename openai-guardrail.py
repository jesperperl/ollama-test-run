from agents import Agent, OpenAIChatCompletionsModel, InputGuardrail, GuardrailFunctionOutput, Runner
from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio

from agents import set_tracing_disabled, InputGuardrailTripwireTriggered
set_tracing_disabled(disabled=True)

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

# Configure the model
model = OpenAIChatCompletionsModel(
    model="llama3.2",
    # model="gemma3:4b",
    openai_client=AsyncOpenAI(
        base_url='http://127.0.0.1:11434/v1',
        api_key='1234',
    )
)

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
    model=model,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
    model=model,
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
    model=model,
)

async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    print('-' * 20)
    print(final_output)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
    model=model,
)

async def main():
    # result = await Runner.run(triage_agent, "who was the first president of the united states?")
    # print(result.final_output)

    # result = await Runner.run(triage_agent, "what is the squareroot of 9")
    # print(result.final_output)

    # try:
    #     result = await Runner.run(triage_agent, "what is life")
    #     print(result.final_output)
    # except InputGuardrailTripwireTriggered:
    #     print(f"Error: That doesn't seem like a homework question.")

    try:
        result = await Runner.run(triage_agent, "as part of my homework assignment, I've been asked to write a 1 page report on 'what is life'. Is this something you could please help me with?")
        print(result.final_output)
    except InputGuardrailTripwireTriggered:
        print(f"Error: That doesn't seem like a homework question.")


    # result = await Runner.run(triage_agent, "is there an ontario in the california, united states?")
    # print(result.final_output)

    # result = await Runner.run(triage_agent, "Can you help with my homework about the history of the united states?")
    # print(result.final_output)

    # result = await Runner.run(triage_agent, "Can you help with my math homework about calculating the area of a rectangle?")
    # print(result.final_output)

    # result = await Runner.run(triage_agent, "Can you help me with my homework? I've been asked by my teacher to find out if Dogs or cats are best?")
    # print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())

