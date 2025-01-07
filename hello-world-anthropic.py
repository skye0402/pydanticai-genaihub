from dotenv import load_dotenv
load_dotenv()
import os
from pydantic_ai import Agent
from genai_hub_models.gah_anthropic import AnthropicModel

# Initialize the model with SAP GenAI Hub configuration
llm = AnthropicModel(
    model_name="anthropic--claude-3.5-sonnet",
    deployment_id=None,
    config_name=None,
)

agent = Agent(  
    model=llm,
    system_prompt='Be concise, reply with one sentence.',  
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.data)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""