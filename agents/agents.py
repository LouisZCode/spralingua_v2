from dotenv import load_dotenv
from langchain.agents import create_agent
from .load_prompts import load_prompts

load_dotenv()

prompts = load_prompts()
test_prompt = prompts["test_prompt"]


# Your agent (unchanged)
_raw_agent = create_agent(
    model="openai:gpt-5-mini",
    system_prompt=test_prompt
)


# Wrapper function for Pipecat compatibility
async def _astream(input_dict, config=None):
    """Translates Pipecat format to agent format and streams response."""
    text = input_dict.get("input", "")
    messages = {"messages": [{"role": "user", "content": text}]}

    async for chunk in _raw_agent.astream(messages, config=config):
        # Handle nested structure: {'model': {'messages': [AIMessage(...)]}}
        if "model" in chunk and "messages" in chunk["model"]:
            for msg in chunk["model"]["messages"]:
                if hasattr(msg, "content") and msg.content:
                    yield msg.content


# Export wrapper that Pipecat can use
class test_agent:
    astream = staticmethod(_astream)
