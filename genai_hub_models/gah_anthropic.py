from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, cast, overload, Union, Literal, Sequence
from json import loads, dumps
from typing_extensions import assert_never

from pydantic_ai import result
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelSettings, EitherStreamedResponse, Model, AgentModel, check_allow_model_requests
from pydantic_ai.usage import Usage
from gen_ai_hub.proxy.native.amazon.clients import Session

try:
    from gen_ai_hub.proxy import get_proxy_client
    from gen_ai_hub.proxy.core.utils import NOT_GIVEN
    from anthropic.types import (
        ContentBlock,
        ContentBlockDeltaEvent,
        Message,
        MessageDeltaEvent,
        MessageParam,
        MessageStartEvent,
        MessageStreamEvent,
    )
except ImportError as _import_error:
    raise ImportError(
        'To use the Anthropic model, you need to install the anthropic package. '
        'You can install it with: pip install anthropic'
    ) from _import_error


AnthropicModelName = Union[str]
"""
Using this more broad type for the model name instead of a strict definition
allows this model to be used more easily with other model types
"""


@dataclass(init=False)
class AnthropicModel(Model):
    """A model that uses the Anthropic API via AWS Bedrock.

    Internally, this uses the GenAI Hub's AWS Bedrock client to interact with Anthropic models.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: AnthropicModelName
    client: Any = field(repr=False)

    def __init__(
        self,
        model_name: str,
        deployment_id: str | None = None,
        config_name: str | None = None,
        proxy_client: Any | None = None,
    ) -> None:
        """Initialize the Anthropic model.

        Args:
            model_name: The name of the model to use
            deployment_id: The deployment ID to use
            config_name: The config name to use
            proxy_client: The proxy client to use
        """
        if proxy_client is None:
            proxy_client = get_proxy_client()

        # Get the deployment
        deployment = proxy_client.select_deployment(
            model_name=model_name,
            deployment_id=deployment_id,
            config_name=config_name,
        )
        if deployment is None:
            raise ValueError(f"No deployment found for model: {model_name}")
        
        # Initialize AWS Bedrock client
        session = Session()
        self.client = session.client(model_name=model_name)
        self.model_name = model_name

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return AnthropicAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f'anthropic:{self.model_name}'

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> dict:
        return {
            'name': f.name,
            'description': f.description,
            'parameters': f.parameters_json_schema,
        }


@dataclass
class AnthropicAgentModel(AgentModel):
    """Implementation of `AgentModel` for Anthropic models via AWS Bedrock."""

    client: Any
    model_name: str
    allow_text_result: bool
    tools: list[dict]

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> tuple[ModelResponse, result.Usage]:
        """Send a request to the Anthropic API.

        Args:
            messages: The messages to send to the API
            model_settings: Optional model settings

        Returns:
            A tuple of (response, usage)
        """
        print(f"Request messages: {messages}")
        response = self._messages_create(messages, False, model_settings)
        return self._process_response(response)

    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> EitherStreamedResponse:
        response = self._messages_create(messages, True, model_settings)
        return self._process_streamed_response(response)

    def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: ModelSettings | None,
    ) -> Message | AsyncIterator[MessageStreamEvent]:
        """Create a message using the Anthropic API via AWS Bedrock.

        Args:
            messages: The messages to send
            stream: Whether to stream the response
            model_settings: Optional model settings

        Returns:
            The API response
        """
        mapped_messages = self._map_messages(messages)
        print(f"Creating message with: {mapped_messages}")
        
        # Convert mapped messages to the format expected by AWS Bedrock
        messages = []
        for msg in mapped_messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Prepare the converse API request
        request_kwargs = {
            "modelId": self.model_name,
            "messages": messages,
            "inferenceConfig": {
                "temperature": 0.7,
                "maxTokens": 1024,
                "topP": 0.999
            }
        }

        # Add tools if they are defined
        if self.tools:
            request_kwargs["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "inputSchema": {
                                "json": loads(dumps(tool["parameters"]))
                            }
                        }
                    }
                    for tool in self.tools
                ]
            }

        print(f"Request kwargs: {request_kwargs}")
        
        # Make the API call through AWS Bedrock
        try:
            response = self.client.converse(**request_kwargs)
            print(f"Response from Claude: {response}")
            return response
            
        except Exception as e:
            print(f"Error calling Claude: {str(e)}")
            raise

    def _process_response(
        self, response: Any
    ) -> tuple[ModelResponse, result.Usage]:
        """Process a response from the Anthropic API.

        Args:
            response: The API response to process

        Returns:
            A tuple of (response, usage)
        """
        print(f"Processing response: {response}")

        # Extract the response content
        output = response.get("output", {})
        message = output.get("message", {})
        content_list = message.get("content", [])
        
        # Create response parts
        parts = []

        # Process each content item
        for content in content_list:
            if "text" in content:
                parts.append(TextPart(content=content["text"]))
            elif "toolUse" in content:
                tool_use = content["toolUse"]
                print(f"Processing tool use: {tool_use}")
                print(f"Tool name: {tool_use['name']}")
                print(f"Tool input: {tool_use['input']}")
                print(f"Tool input type: {type(tool_use['input'])}")
                
                tool_part = ToolCallPart.from_raw_args(
                    tool_use["name"],
                    dumps(tool_use["input"]),  # Convert dict to JSON string
                    tool_use.get("toolUseId", ""),
                )
                print(f"Created ToolCallPart: {tool_part}")
                print(f"ToolCallPart args: {tool_part.args}")
                print(f"ToolCallPart args type: {type(tool_part.args)}")
                parts.append(tool_part)

        # Create usage metrics
        usage = Usage(
            requests=1,
            request_tokens=output.get("usage", {}).get("inputTokens", 0),
            response_tokens=output.get("usage", {}).get("outputTokens", 0),
            total_tokens=output.get("usage", {}).get("totalTokens", 0),
            details={},
        )

        return ModelResponse(parts=parts), usage

    def _map_messages(self, messages: list[ModelMessage]) -> list[dict[str, Any]]:
        """Map a list of model messages to a list of Anthropic messages.

        Args:
            messages: The messages to map

        Returns:
            A list of Anthropic messages
        """
        mapped_messages = []
        current_role = "user"  # Track the current role
        tool_results_buffer = []  # Buffer to collect tool results
        
        for message in messages:
            print(f"Mapping message: {message}")
            print(f"Message type: {type(message)}")
            print(f"Message parts: {message.parts}")

            if isinstance(message, ModelRequest):
                for part in message.parts:
                    print(f"Processing part: {part}")
                    if isinstance(part, SystemPromptPart):
                        # Add system message as user instruction
                        mapped_messages.append({
                            "role": "user",
                            "content": [{"text": f"Instructions: {part.content}"}]
                        })
                        # Add assistant acknowledgment
                        mapped_messages.append({
                            "role": "assistant", 
                            "content": [{"text": "I understand and will follow these instructions."}]
                        })
                        current_role = "user"  # Next message should be user
                    elif isinstance(part, UserPromptPart):
                        # Add user message
                        if current_role == "assistant":
                            # If last message was assistant, this can be user
                            mapped_messages.append({
                                "role": "user",
                                "content": [{"text": part.content}]
                            })
                            current_role = "user"
                        else:
                            # If last message was user, combine with previous
                            if mapped_messages and mapped_messages[-1]["role"] == "user":
                                mapped_messages[-1]["content"].append({"text": part.content})
                            else:
                                mapped_messages.append({
                                    "role": "user",
                                    "content": [{"text": part.content}]
                                })
                    elif isinstance(part, ToolReturnPart):
                        # Collect tool results in the buffer
                        tool_results_buffer.append({
                            "toolResult": {
                                "toolUseId": part.tool_call_id,
                                "content": [{"text": dumps(part.content)}],
                                "status": "success"
                            }
                        })

                # If we have collected tool results, add them all in one message
                if tool_results_buffer:
                    if current_role == "assistant":
                        # If last message was assistant, add as user
                        mapped_messages.append({
                            "role": "user",
                            "content": tool_results_buffer
                        })
                        current_role = "user"
                    else:
                        # If last message was user, add as assistant
                        mapped_messages.append({
                            "role": "assistant",
                            "content": tool_results_buffer
                        })
                        current_role = "assistant"
                    tool_results_buffer = []  # Clear the buffer

            elif isinstance(message, ModelResponse):
                # Process response parts
                content_list = []
                for part in message.parts:
                    if isinstance(part, TextPart):
                        if part.content != "":
                            content_list.append({"text": part.content})
                    elif isinstance(part, ToolCallPart):
                        content_list.append({
                            "toolUse": {
                                "toolUseId": part.tool_call_id,
                                "name": part.tool_name,
                                "input": loads(part.args_as_json_str())
                            }
                        })
                if content_list:
                    if current_role == "user":
                        # If last message was user, add as assistant
                        mapped_messages.append({
                            "role": "assistant",
                            "content": content_list
                        })
                        current_role = "assistant"
                    else:
                        # If last message was assistant, add as user
                        mapped_messages.append({
                            "role": "user",
                            "content": content_list
                        })
                        current_role = "user"

        # Ensure conversation ends with a user message
        if mapped_messages and mapped_messages[-1]["role"] == "assistant":
            mapped_messages.append({
                "role": "user",
                "content": [{"text": "Please continue."}]
            })
            current_role = "user"

        return mapped_messages

    @staticmethod
    def _process_streamed_response(response: Any) -> EitherStreamedResponse:
        raise NotImplementedError("Streaming is not yet supported for Anthropic models via AWS Bedrock")


def _map_tool_call(t: ToolCallPart) -> dict:
    return {
        'type': 'function',
        'function': {
            'name': t.name,
            'arguments': t.arguments,
        },
        'id': t.id,
    }