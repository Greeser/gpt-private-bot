import io
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import discord
import aiohttp
from src.base import Message
from src.utils import split_into_shorter_messages, logger, close_thread, discord_image
from src.constants import OPENAI_API_KEY, OPENAI_API_URL, OPENAI_MODEL, MAX_CHARS_PER_REPLY_MSG, DALLE_API_URL, DALLE_MODEL


class CompletionResult(Enum):
    OK = 0
    TOO_LONG = 1
    ERROR = 2


@dataclass
class CompletionData:
    status: CompletionResult
    reply_text: Optional[str]
    status_text: Optional[str]

@dataclass
class ImageCompletionData:
    status: CompletionResult
    image64: Optional[str]
    status_text: Optional[str]

async def generate_completion_response(
    messages: List[Message],
) -> CompletionData:
    try:
        async with aiohttp.ClientSession() as session:
            messages = [message.render() for message in messages]
            async with session.post(
                    url=OPENAI_API_URL,
                    json={
                        'model': OPENAI_MODEL,
                        'messages': messages
                    },
                    auth=aiohttp.BasicAuth("", OPENAI_API_KEY)
                    ) as r:
                if r.status == 200:
                    js = await r.json()
                    reply = js['choices'][0]['message']['content']
                    return CompletionData(status=CompletionResult.OK, reply_text=reply, status_text=None)
                else:
                    js = await r.json()
                    code = js['error']['code']
                    status = CompletionResult.TOO_LONG if code == 'context_length_exceeded' else CompletionResult.ERROR
                    return CompletionData(status=status, reply_text=None, status_text=js)
    except Exception as e:
        logger.exception(e)
        return CompletionData(
            status=CompletionResult.ERROR, reply_text=None, status_text=str(e)
        )

async def generate_image_response(
    prompt: Message,
) -> CompletionData:
    try:
        async with aiohttp.ClientSession() as session:
            prompt = prompt.render().get('content', '')
            async with session.post(
                    url=DALLE_API_URL,
                    json={
                        'prompt': prompt,
                        'n': 1,
                        'size': '256x256',
                        'response_format': 'b64_json',
                        'model': DALLE_MODEL
                    },
                    auth=aiohttp.BasicAuth("", OPENAI_API_KEY)
                    ) as r:
                if r.status == 200:
                    js = await r.json()
                    reply = js['data'][0]['b64_json']
                    return ImageCompletionData(status=CompletionResult.OK, image64=reply, status_text=None)
                else:
                    js = await r.json()
                    code = js['error']['code']
                    status = CompletionResult.TOO_LONG if code == 'context_length_exceeded' else CompletionResult.ERROR
                    return ImageCompletionData(status=status, image64=None, status_text=js)
    except Exception as e:
        logger.exception(e)
        return ImageCompletionData(
            status=CompletionResult.ERROR, image64=None, status_text=str(e)
        )


async def process_response(
    thread: discord.Thread, response_data: CompletionData
):
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text
    if status is CompletionResult.OK:
        if not reply_text:
            await thread.send(
                embed=discord.Embed(
                    description=f"**Invalid response** - empty response",
                    color=discord.Color.yellow(),
                )
            )
        else:
            shorter_response = split_into_shorter_messages(reply_text)
            for r in shorter_response:
                if len(r) > MAX_CHARS_PER_REPLY_MSG:
                    file = discord.File(io.StringIO(r), f'message.txt')
                    await thread.send(file=file)
                else:
                    await thread.send(r)
    elif status is CompletionResult.TOO_LONG:
        await close_thread(thread)
    else:
        await thread.send(
            embed=discord.Embed(
                description=f"**Error** - {status_text}",
                color=discord.Color.yellow(),
            )
        )

async def process_image_response(
    channel: discord.TextChannel,
    response_data: ImageCompletionData
):
    status = response_data.status
    image64 = response_data.image64
    status_text = response_data.status_text
    if status is CompletionResult.OK:
        if not image64:
            await channel.send(
                embed=discord.Embed(
                    description=f"**Invalid response** - empty response",
                    color=discord.Color.yellow(),
                )
            )
        else:
            file = discord_image(image64)
            await channel.send(file=file)
    else:
        await channel.send(
            embed=discord.Embed(
                description=f"**Error** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
