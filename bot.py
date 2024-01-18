import io
import os

import cv2
import dlib
from pydub import AudioSegment

import numpy as np
import matplotlib.pyplot as plt
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

# DEBUG_MODE using for show founded faces using matplotlib.pyplot
DEBUG_MODE = False

client = TelegramClient('bot_session', api_id, api_hash).start(bot_token=bot_token)


# Voice messages handler
@client.on(events.NewMessage(func=lambda event: event.voice))
async def handle_voice_message(event):
    sender_id = event.sender_id
    folder_path = f"Storage/{sender_id}/audio"
    # Creating folder for downloads if not exist
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except Exception as e:
        print(f"Error creating folder: {e}")
    # Counting files for assign file number // change this method if files could be moved or replaced
    files = os.listdir(folder_path)
    output_path = f'{folder_path}/audio_message_{len(files)}.wav'
    try:
        await convert_to_wav(event, output_path)
    except Exception as e:
        print(f"Error handling voice message: {e}")


async def convert_to_wav(event, output_path, sample_rate=16000):
    try:
        # Working with bytes for save file in specific path
        audio_data = await event.download_media(bytes)
        # Creating a file-like object from the raw audio data
        audio_file = io.BytesIO(audio_data)
        # Using AudioSegment.from_file with the file-like object
        audio = AudioSegment.from_file(audio_file, format='ogg')
        # Setting rate
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio.export(output_path, format="wav")
    except Exception as e:
        print(f"Error converting to WAV: {e}")


# Photo handler
@client.on(events.NewMessage(func=lambda event: event.photo))
async def handle_photo(event):
    sender_id = event.sender_id
    # working with bytes for save file in specific path
    image_bytes = await event.download_media(bytes)

    if await has_faces(image_bytes):
        try:
            folder_path = f"Storage/{sender_id}/photo"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            # Counting files for assign file number // change this method if files could be moved or replaced
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
    ''' Function using 2 detection methods:
    - dlib.get_frontal_face_detector()
    - Haar's cascade
    '''
    try:
        # Preparing image for detection
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Trying to detect
        faces = dlib_basic_detector(gray_image)
        if not faces:
            # Trying to use haar cascade
            faces = haarcascade_detector(gray_image)

        if faces is not None and len(faces) > 0:
            if DEBUG_MODE:
                show_faces(image, gray_image, faces)
            return True

    except Exception as e:
        print("Exception raised:", e)
    return False


def haarcascade_detector(gray_image):
    face_cascade = cv2.CascadeClassifier("Models/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.04, minNeighbors=5)
    if faces is not None:
        print(f"{len(faces)} faces found with Haar's cascade")
    return faces


def dlib_basic_detector(gray_image):
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray_image, 1)

    # detector returns <dlib> object with rectangles with founded faces o empty object
    # so lets transform it to coordinates
    faces_coordinates = [(face.left(), face.top(), face.width(), face.height()) for face in faces]

    if faces_coordinates:
        print(f"{len(faces_coordinates)} faces found with dlib_basic_detector")

    return faces_coordinates


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def show_faces(image, gray_image, faces):
    if len(faces) > 0:
        predictor = dlib.shape_predictor('Models/shape_predictor_68_face_landmarks.dat')
        for x, y, w, h in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # creating the rectangle object from the outputs of haar cascade calssifier
            drect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = predictor(gray_image, drect)
            points = shape_to_np(landmarks)
            for i in points:
                x = i[0]
                y = i[1]
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
    client.start()
    client.run_until_disconnected()
