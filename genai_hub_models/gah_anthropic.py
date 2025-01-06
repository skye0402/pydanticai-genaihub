from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, cast, overload, Union, Literal
from json import dumps, loads
from typing import Any, AsyncIterator, cast

from pydantic_ai import messages, result
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelSettings, EitherStreamedResponse, Model, AgentModel, check_allow_model_requests
from pydantic_ai.usage import Usage
from typing_extensions import assert_never

from httpx import AsyncClient as AsyncHTTPClient
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
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.parameters_json_schema,
            },
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
        response = self._messages_create(messages[-1], False, model_settings)
        return self._process_response(response)

    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> EitherStreamedResponse:
        response = self._messages_create(messages[-1], True, model_settings)
        return self._process_streamed_response(response)

    @staticmethod
    def _map_message(message: ModelMessage) -> dict | list[dict]:
        """Map a ModelMessage to an Anthropic message format."""
        print(f"Mapping message: {message}")
        print(f"Message type: {type(message)}")
        if isinstance(message, ModelRequest):
            print(f"Message parts: {message.parts}")
            mapped_messages = []
            for part in message.parts:
                print(f"Processing part: {part}")
                if isinstance(part, SystemPromptPart):
                    mapped_messages.append({'role': 'assistant', 'content': part.content})
                elif isinstance(part, UserPromptPart):
                    mapped_messages.append({'role': 'user', 'content': part.content})
                elif isinstance(part, TextPart):
                    mapped_messages.append({'role': 'assistant', 'content': part.content})
                elif isinstance(part, ToolCallPart):
                    mapped_messages.append({
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [_map_tool_call(part)],
                    })
                elif isinstance(part, ToolReturnPart):
                    mapped_messages.append({
                        'role': 'assistant',
                        'content': part.content,
                    })
                elif isinstance(part, RetryPromptPart):
                    mapped_messages.append({'role': 'user', 'content': part.content})
                else:
                    assert_never(part)
            print(f"Mapped messages: {mapped_messages}")
            return mapped_messages
        elif isinstance(message, ModelResponse):
            return {'role': 'assistant', 'content': message.content}
        else:
            assert_never(message)
    
    def _messages_create(
        self,
        message: ModelMessage,
        stream: bool,
        model_settings: ModelSettings | None,
    ) -> Any:
        """Create a chat completion request to the Anthropic API.

        Args:
            message: The message to send to the API
            stream: Whether to stream the response
            model_settings: Optional model settings

        Returns:
            The response from the API
        """
        mapped = self._map_message(message)
        request_messages = mapped if isinstance(mapped, list) else [mapped]
        
        request_body = {
            'messages': request_messages,
            'max_tokens': 1000,  # Required by AWS Bedrock
            'anthropic_version': 'bedrock-2023-05-31',  # Required by AWS Bedrock
        }
        
        if model_settings is not None:
            if model_settings.temperature is not None:
                request_body['temperature'] = model_settings.temperature
            if model_settings.max_tokens is not None:
                request_body['max_tokens'] = model_settings.max_tokens
        
        print(f"Request body: {request_body}")
        response = self.client.invoke_model(
            body=dumps(request_body).encode(),
            modelId=self.model_name,
        )
        
        if stream:
            return response['body']
        else:
            response_bytes = response['body'].read()
            response_data = loads(response_bytes.decode())
            print(f"Response data: {response_data}")
            return response_data

    @staticmethod
    def _process_response(response: dict) -> tuple[ModelResponse, result.Usage]:
        """Process a non-streamed response, and prepare a message to return."""
        content = response.get('content', [])
        if isinstance(content, list) and len(content) > 0:
            content = content[0].get('text', '')
        
        usage_data = response.get('usage', {})
        usage = Usage()
        usage.requests = 1
        usage.request_tokens = usage_data.get('input_tokens', 0)
        usage.response_tokens = usage_data.get('output_tokens', 0)
        usage.total_tokens = usage.request_tokens + usage.response_tokens if usage.request_tokens is not None else None
        
        return ModelResponse.from_text(content), usage

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