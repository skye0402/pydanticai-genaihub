# PydanticAI GenAI Hub

A Python library providing Pydantic-based model implementations for various LLM providers.
It's intended to be used with SAP's Generative AI Hub SDK.

This project is not affiliated with SAP.

## Installation

You can install the package with pip:

```bash
pip install pydanticai-genai-hub
```

To install with specific provider support:

```bash
# For Anthropic support
pip install "pydanticai-genai-hub[anthropic]"

# For OpenAI support
pip install "pydanticai-genai-hub[openai]"

# For all providers
pip install "pydanticai-genai-hub[all]"
```

## Usage

### Anthropic Model

```python
from pydanticai_genai_hub.anthropic import AnthropicModel

model = AnthropicModel(
    model_name="anthropic--claude-3.5-sonnet",
)
```

### OpenAI Model

```python
from pydanticai_genai_hub.openai import OpenAIModel

model = OpenAIModel(
    model_name="gpt-4o",
)
```

### 'Hello World' Example with AWS Bedrock Anthropic

```python
from pydantic_ai import Agent
from pydanticai_genai_hub.anthropic import AnthropicModel

# Initialize the model with SAP GenAI Hub configuration
llm = AnthropicModel(
    model_name="anthropic--claude-3.5-sonnet",
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
```

## License

MIT License
