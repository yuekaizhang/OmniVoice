from omnivoice import OmniVoice
import torch
import torchaudio

model = OmniVoice.from_pretrained(
    "/workspace_yuekai/HF/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)
# Apple Silicon users: use device_map="mps" instead

audio = model.generate(
    text="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。",
    ref_audio="prompt_audio.wav",
    ref_text="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
) # audio is a list of `torch.Tensor` with shape (1, T) at 24 kHz.

# If you don't want to input `ref_text` manually, you can directly omit the `ref_text`.
# The model will use Whisper ASR to auto-transcribe it.

torchaudio.save("out.wav", audio[0], 24000)