from json import load
from dotenv import load_dotenv
from langchain.agents import create_agent
from .load_prompts import load_prompts

load_dotenv()

prompts = load_prompts()
test_prompt = prompts["test_prompt"]
print(test_prompt)


test_agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=test_prompt
)
