"""
Here we will contruct the dynamic prompts that will be given to the AI Dynamically.
"""
from .load_prompts import load_prompts
prompts = load_prompts()
conversation_prompt = prompts["conversationalist_prompt"]

from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class Context:
    #This will come from the database in the future:
    user_name : str = "Luis"
    topic : str = "football"
    user_level: str = "A1"




@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    level_config = prompts["user_level"][ctx.user_level]
    #These ones maybe to come from the user selection in the UI? specially the last []
    agent_story = prompts["agent_story"]["happy_harry"]
    agent_personality = prompts["agent_personality"]["happy"]

    return conversation_prompt.format(
        name = ctx.user_name,
        user_level = level_config,
        topic = ctx.topic,
        agent_story = agent_story,
        agent_personality = agent_personality
    )
