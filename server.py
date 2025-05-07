import pyaudio
import numpy as np
import asyncio
import websockets
import json
from openwakeword.model import Model
import argparse
import time
import signal

shutdown_event = asyncio.Event()
# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--chunk_size", type=int, default=2048)
# parser.add_argument("--model_path", type=str, default="")
parser.add_argument("--inference_framework", type=str, default='onnx')  # use onnx to avoid tflite issues
args = parser.parse_args()

# Setup audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load model
owwModel = Model(
    ['alexa', 'hey_jarvis']
    # wakeword_models=[args.model_path] if args.model_path else None,
    # inference_framework=args.inference_framework
)

# Set of connected clients
clients = set()

async def notify_clients(message):
    if clients:  # only send if someone is listening
        await asyncio.gather(*(client.send(json.dumps(message)) for client in clients))

async def wakeword_loop():
    # Add this before your loop
    last_detection_time = 0
    cooldown_seconds = 2  # Only send one detection per 2 seconds
    print("Wake word detection started. Waiting for connections...")

    while not shutdown_event.is_set():
        audio_chunk = np.frombuffer(mic_stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        prediction = owwModel.predict(audio_chunk)

        detected = any(scores[-1] > 0.5 for scores in owwModel.prediction_buffer.values())
        current_time = time.time()

        if detected and (current_time - last_detection_time > cooldown_seconds):
            last_detection_time = current_time
            print("Wakeword detected!")
            # Send to WebSocket clients here
            await notify_clients({"message": "wakeword_detected"})
            # for ws in clients:
                # await ws.send("wakeword_detected")
                # await notify_clients({"message": "wakeword_detected"})

        await asyncio.sleep(0.001)  # yield to event loop

async def handler(websocket):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

def shutdown_handler():
    print("\nReceived shutdown signal. Cleaning up...")
    shutdown_event.set()

signal.signal(signal.SIGINT, lambda s, f: shutdown_handler())  # CTRL+C
signal.signal(signal.SIGTERM, lambda s, f: shutdown_handler())  # kill command


async def main():
    server = await websockets.serve(handler, "localhost", 9091)
    print("WebSocket server running on ws://localhost:9091")

    wakeword_task = asyncio.create_task(wakeword_loop())
    await shutdown_event.wait()  # Wait for shutdown signal

    print ("Shutting down...")

    wakeword_task.cancel()  # Cancel the wakeword loop task

    try:
        await wakeword_task

    except asyncio.CancelledError:
        print("Wakeword loop cancelled.")
        pass
    # await asyncio.gather(server.wait_closed(), wakeword_loop())
    # close audio sources
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()

    # close websocket server
    server.close()
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())

