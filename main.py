from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

comedian_prompt = "You are a full stack comedian"

agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=comedian_prompt
)

message = "tell me a joke about mexico city"

result = agent.invoke(
    {"role": "user", "content": message}
)

for i, msg in enumerate(result["messages"]):
    msg.pretty_print()