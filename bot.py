import io
import os

import cv2
import dlib
from pydub import AudioSegment

import numpy as np
from telethon import TelegramClient, events

from dotenv import load_dotenv

load_dotenv(".env")

import logging
logging.basicConfig(format='[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s',
                    level=logging.WARNING)

api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
bot_token = os.getenv("BOT_TOKEN")

bot_name = 'my_shiny_telegram_bot'
bot_address = 't.me/my_shiny_telegram_bot'

client = TelegramClient('bot_session', api_id, api_hash).start(bot_token=bot_token)


# Photo handler
@client.on(events.NewMessage(func=lambda event: event.photo))
async def handle_photo(event):
    sender_id = event.sender_id
    image_bytes = await event.download_media(bytes)

    if await has_faces(image_bytes):
        try:
            folder_path = f"Storage/{sender_id}/photo"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # Counting files for create file number
            files = os.listdir(folder_path)
            # Saving photo
            with open(f'{folder_path}/photo_{len(files)}.jpg', 'wb') as photo:
                photo.write(image_bytes)
                await event.reply(f'The image containing faces and saved')
        except Exception as e:
            print(f"Error creating folder: {e}")
    else:
        await event.reply(f'The image does not contain faces and will be skipped')


async def has_faces(image_bytes):

    try:
        detector = dlib.get_frontal_face_detector()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_image, 1)
        print(f"{len(faces)} faces found")
        if len(faces) > 0:
            return True
    except Exception as e:
        print("Exception raised:", e)
    return False


# Voice messages handler
@client.on(events.NewMessage(func=lambda event: event.voice))
async def handle_voice_message(event):
    sender_id = event.sender_id
    folder_path = f"Storage/{sender_id}/audio"

    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except Exception as e:
        print(f"Error creating folder: {e}")

    files = os.listdir(folder_path)
    output_path = f'{folder_path}/audio_message_{len(files)}.wav'
    try:
        await convert_to_wav(event, output_path)
    except Exception as e:
        print(f"Error handling voice message: {e}")


async def convert_to_wav(event, output_path, sample_rate=16000):
    try:
        audio_data = await event.download_media(bytes)

        # Create a file-like object from the raw audio data
        audio_file = io.BytesIO(audio_data)

        # Use AudioSegment.from_file with the file-like object
        audio = AudioSegment.from_file(audio_file, format='ogg')

        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio.export(output_path, format="wav")

    except Exception as e:
        print(f"Error converting to WAV: {e}")


if __name__ == '__main__':
    client.start()
    client.run_until_disconnected()
