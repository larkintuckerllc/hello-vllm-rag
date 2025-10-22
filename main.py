from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city.
    
    Args:
        city: The city to get the weather for
    """
    return f"It's always sunny in {city}!"

def main():
    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        max_tokens=1000
    )
    agent = create_agent(
        model=llm,
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    final_message = result["messages"][-1]
    print(final_message.content)

if __name__ == "__main__":
    main()
