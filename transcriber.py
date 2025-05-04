import whisper

model = whisper.load_model("base", device="cpu")

def transcribe_audio(path):
    result = model.transcribe(path)
    return result["text"]
