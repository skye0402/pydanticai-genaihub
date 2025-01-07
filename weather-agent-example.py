from __future__ import annotations as _annotations

from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
from dataclasses import dataclass
from typing import Any
import time

import logfire
from devtools import debug
from httpx import AsyncClient

from pydantic_ai import Agent, ModelRetry, RunContext

from genai_hub_models.gah_anthropic import AnthropicModel
from genai_hub_models.gah_openai import OpenAIModel
# Initialize the model with SAP GenAI Hub configuration
llm_a = AnthropicModel(
    model_name="anthropic--claude-3.5-sonnet",
    deployment_id=None,
    config_name=None,
)

llm_o = OpenAIModel(
    model_name="gpt-4o",
)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')


@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None


weather_agent = Agent(
    model=llm_o,
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    system_prompt=(
        'Be concise, reply with one sentence.'
        'Use the `get_lat_lng` tool to get the latitude and longitude of the locations, '
        'then use the `get_weather` tool to get the weather.'
    ),
    deps_type=Deps,
    retries=2,
)


# Add a semaphore for rate limiting
geocode_semaphore = asyncio.Semaphore(1)  # Allow only 1 request at a time

@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    if ctx.deps.geo_api_key is None:
        # if no API key is provided, return a dummy response (London)
        return {'lat': 51.1, 'lng': -0.1}

    params = {
        'q': location_description,
        'api_key': ctx.deps.geo_api_key,
    }
    
    async with geocode_semaphore:  # This ensures only one request at a time
        with logfire.span('calling geocode API', params=params) as span:
            r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params, timeout=30.0)
            r.raise_for_status()
            data = r.json()
            span.set_attribute('response', data)
            # Add a 1-second delay after each request
            await asyncio.sleep(1)

        if data:
            return {'lat': float(data[0]['lat']), 'lng': float(data[0]['lon'])}
        else:
            raise ModelRetry('Could not find the location')


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.weather_api_key is None:
        # if no API key is provided, return a dummy response
        return {'temperature': '21 °C', 'description': 'Sunny'}

    params = {
        'apikey': ctx.deps.weather_api_key,
        'location': f'{lat},{lng}',
        'units': 'metric',
    }
    try:
        with logfire.span('calling weather API', params=params) as span:
            r = await ctx.deps.client.get(
                'https://api.tomorrow.io/v4/weather/realtime',
                params=params,
                timeout=30.0  # Add 30 second timeout
            )
            r.raise_for_status()
            data = r.json()
            span.set_attribute('response', data)

        values = data['data']['values']
        # https://docs.tomorrow.io/reference/data-layers-weather-codes
        code_lookup = {
            1000: 'Clear, Sunny',
            1100: 'Mostly Clear',
            1101: 'Partly Cloudy',
            1102: 'Mostly Cloudy',
            1001: 'Cloudy',
            2000: 'Fog',
            2100: 'Light Fog',
            4000: 'Drizzle',
            4001: 'Rain',
            4200: 'Light Rain',
            4201: 'Heavy Rain',
            5000: 'Snow',
            5001: 'Flurries',
            5100: 'Light Snow',
            5101: 'Heavy Snow',
            6000: 'Freezing Drizzle',
            6001: 'Freezing Rain',
            6200: 'Light Freezing Rain',
            6201: 'Heavy Freezing Rain',
            7000: 'Ice Pellets',
            7101: 'Heavy Ice Pellets',
            7102: 'Light Ice Pellets',
            8000: 'Thunderstorm',
        }
        return {
            'temperature': f'{values["temperatureApparent"]:0.0f}°C',
            'description': code_lookup.get(values['weatherCode'], 'Unknown'),
        }
    except Exception as e:
        print(f"Warning: Weather API request failed: {str(e)}. Using fallback data.")
        return {'temperature': '21 °C', 'description': 'Sunny (Fallback)'}


async def main():
    async with AsyncClient() as client:
        # create a free API key at https://www.tomorrow.io/weather-api/
        weather_api_key = os.getenv('WEATHER_API_KEY')
        # create a free API key at https://geocode.maps.co/
        geo_api_key = os.getenv('GEO_API_KEY')
        deps = Deps(
            client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key
        )
        result = await weather_agent.run(
            'What is the weather like in London and in Wiltshire?', deps=deps
        )
        debug(result)
        print('Response:', result.data)


if __name__ == '__main__':
    asyncio.run(main())