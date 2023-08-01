#! python3.8

import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import sounddevice as sd
import soundfile as sf
import openai
from elevenlabs import set_api_key, generate, VOICES_CACHE, play
from dotenv import load_dotenv

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from collections import namedtuple

DEFAULT_OPENAI_PROMPT = """You are a participant in a dialogue, respond in a short sentence to the participant.
Use an active tone, answer the participant in the most interesting way, addressing any mentioned points,
and try to continue the conversation. Don't say irrelevant things.
if the dialog does not make sense or is not a proper sentence or unclear or if there are no main points, return empty string.
"""


def play_reply(text, device_idx, voice_idx=3):
    audio_bytes = generate(text=text, voice=VOICES_CACHE[voice_idx])
    sd.default.device = device_idx
    sd.play(*sf.read(io.BytesIO(audio_bytes)))
    sd.wait()


def get_reply(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": DEFAULT_OPENAI_PROMPT
        },
            {
            "role": "user",
            "content": text
        }],
        temperature=0,
        max_tokens=2000
    )
    return response.choices[0]["message"]["content"]


def print_available_devices():
    device_list = sd.query_devices()
    print("Available audio devices:")
    for idx, device in enumerate(device_list):
        print(f"Device {idx}:")
        print(f"  Name: {device['name']}")
        print(f"  Default Sample Rate: {device['default_samplerate']:.2f} Hz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    args = parser.parse_args()

    print_available_devices()
    input_device_idx = int(input('Input Device Index:'))
    output_device_idx = int(input('Ouput Device Index:'))

    # Args = namedtuple(
    #     'Args', 'model non_english energy_threshold record_timeout phrase_timeout')
    # args = Args('small', False, 1000, 2, 3)

    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # output of computer to multi-output virtual device (Discord Audio) goes to blackhole + earphones
    # use the blackhole virtual device (0) `python -m sounddevice -h`
    source = sr.Microphone(sample_rate=16000, device_index=input_device_idx)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    last_sample += data_queue.get()

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(
                    last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(
                    temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                    # print(transcription)
                    input_text = '\n'.join(transcription)
                    print("<<< " + input_text + '\n')
                    reply = get_reply(input_text)
                    print(">>> " + reply + '\n\n')
                    if (reply):
                        transcription = ['']
                        play_reply(reply, output_device_idx)
                else:
                    transcription[-1] = text
                print(len(transcription))
                sleep(0.25)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    load_dotenv()
    if not os.getenv("ELEVEN_API_KEY"):
        print("Please set ELEVEN_API_KEY in env")
    elif not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in env")
    else:
        set_api_key(os.getenv("ELEVEN_API_KEY"))
        main()
