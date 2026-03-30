import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio
import time
from openai import OpenAI

def audio_parse(path_to_audio: str):
    model_path = "./cohere-transcribe-local"

    # 1. Load processor and model
    processor = AutoProcessor.from_pretrained(model_path)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("mps")

    # 2. Load audio at 16kHz using Hugging Face's utility
    audio = load_audio(path_to_audio, sampling_rate=16000)
    duration_s = len(audio) / 16000
    print(f"Audio duration: {duration_s / 60:.1f} minutes")

    # 3. Process the audio (automatic chunking happens here)
    inputs = processor(
        audio=audio,
        sampling_rate=16000,
        return_tensors="pt",
        language="en",
        punctuation=True  # Set to False for lowercase without punctuation
    )

    # Get the chunk index for reassembly
    audio_chunk_index = inputs.get("audio_chunk_index")

    # 4. Move inputs to device
    inputs.to(model.device, dtype=model.dtype)

    # 5. Generate transcription
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,  # Adjust based on expected output length
            temperature=0.7      # Locks it to the sharpest probabilities
        )

    # 6. Decode with chunk reassembly
    text = processor.decode(
        outputs,
        skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index,
        language="en"
    )[0]

    elapsed = time.time() - start
    print(f"\nTranscribed in {elapsed:.1f}s")
    print(f"RTFx: {duration_s / elapsed:.1f}")
    print(f"\n=== Full Transcription ({len(text.split())} words) ===")
    print(text)

    return text


def validation(text: str) -> str:
    """Send transcribed text to OpenAI API for validation/processing."""
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="dummy",  # Local server may not need a real key
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Adjust based on what's running on localhost:8001
        messages=[
            {
                "role": "system",
                "content": "You are a lyrics transcription corrector. Your only job is to correct any spelling, grammar, or transcription errors in the lyrics. Do NOT analyze, interpret, explain, or add commentary about the meaning of the lyrics. Do NOT make up content that isn't there. Just return the corrected text with proper punctuation and formatting."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.9,  # Higher temperature for more creative responses
    )

    result = response.choices[0].message.content
    print(f"\n=== Validated/Processed Text ===")
    print(result)

    return result




if __name__ == "__main__":
    audio_path = '/Users/zacharyaldin/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings/20260330 131345.m4a'
    text = audio_parse(audio_path)
    validation(text)