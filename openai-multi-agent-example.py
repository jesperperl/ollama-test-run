from duckduckgo_search import DDGS
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool
from openai import AsyncOpenAI
from datetime import datetime

from agents import set_tracing_disabled
set_tracing_disabled(disabled=True)


# Get the current date in YYYY-MM format
current_date = datetime.now().strftime('%Y-%m')

# Configure the model
model = OpenAIChatCompletionsModel(
    model="llama3.2",
    openai_client=AsyncOpenAI(
        base_url='http://127.0.0.1:11434/v1',
        api_key='1234',
    )
)

# Configure the gemma model
# model_gemma = OpenAIChatCompletionsModel(
#     model="gemma3:4b",
#     openai_client=AsyncOpenAI(
#         base_url='http://127.0.0.1:11434/v1',
#         api_key='1234',
#     )
# )

@function_tool
def get_news_articles(topic):
    print(f"Running DuckDuckGo news search for {topic}...")

    # DuckDuckGo Search
    ddg_api = DDGS()
    results = ddg_api.text(f"{topic} {current_date}", max_results=5)
    if results:
        news_results = '\n\n'.join([f"Title: {result['title']}\nURL: {result['href']}\nDescription: {result['body']}" for result in results])
        print(news_results)
        return news_results
    else:
        return f"Could not find news results for {topic}"

news_agent = Agent(
    name="News Assistant",
    instructions="You provide the lastest news articles for a given topic using DuckDuckGo Search.",
    tools=[get_news_articles],
    model=model,
)

editor_agent = Agent(
    name="Editor assistant",
    instructions="Rewrite and give me as news article ready for publishing. Each News story in separate section.",
    model=model,
)

def run_news_workflow(topic):
    print("Running news Agent workflow...")

    # Step 1: Fetch news
    news_response = Runner.run_sync(
        news_agent,
        f"Get me the news about {topic} on {current_date}",
    )

    # Access the content for RunResult object
    raw_news = news_response.final_output

    # Step 2: Pass news to editor for final review
    edited_news_response = Runner.run_sync(
        editor_agent,
        raw_news,
    )

    # Access the content from RunResult object
    edited_news = edited_news_response.final_output

    print("Final news article:")
    print(edited_news)

    return edited_news

# print(run_news_workflow("AI"))
# print(run_news_workflow("Donald Trump"))
# print(run_news_workflow("Denmark"))
# print(run_news_workflow("DOJ"))
# print(run_news_workflow("OpenAI"))
print(run_news_workflow("IT jobs"))
