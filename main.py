import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio
import time

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




if __name__ == "__main__":
    audio_parse('/Users/zacharyaldin/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings/20260330 131345.m4a')