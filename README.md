# HelloChat

Have audio chats be handled by AI.

Audio apps like Discord / Zoom have an output and input, which becomes our input and ouput respectively.

Output from the app is input to our 1st device, which transcribes the audio from the app. The response is generated with Open AI, then using ElevenLabs text2speech, the result is played to our 2nd device, as though we were speaking into the input microphone of Discord.

## Virtual Audio Devices

A simple way to do this is to download [Blackhole](https://existential.audio/blackhole/) twice, e.g. install both 2ch and 16ch devices.

## API Keys

Create `.env` file with keys in `.env.example`

## Running the app

- Install dependencies `pip install -r requirements.txt`
- `python main.py`
  - set the device numbers of the input (output of app) and output (input of app)
