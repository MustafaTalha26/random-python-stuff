import os

from discord import FFmpegPCMAudio
from discord.ext.commands import Bot
from discord import Intents
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
PREFIX = os.getenv('DISCORD_PREFIX')

client = Bot(command_prefix=list(PREFIX),intents=Intents.default())


@client.event
async def on_ready():
    print('Music Bot Ready')


@client.command(aliases=['p', 'pla'])
async def play(ctx, url: str = 'http://stream.radioparadise.com/rock-128'):
    channel = ctx.message.author.voice.channel
    global player
    try:
        player = await channel.connect()
    except:
        pass
    player.play(FFmpegPCMAudio('http://stream.radioparadise.com/rock-128'))


@client.command(aliases=['s', 'sto'])
async def stop(ctx):
    player.stop()


client.run(TOKEN)