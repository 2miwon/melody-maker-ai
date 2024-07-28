from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
import torchaudio
import scipy
import torch

processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
# device_name = "cpu"
# device_name = "mps"
# device = torch.device(device_name)
# model.to(device)
# torchaudio.set_audio_backend("sox_io")

def combine_midi(video_length, input_video_path, path, description):
    melody, sr = torchaudio.load(f"{path}/melody.wav")
    inputs = processor(
            audio=melody[0],
            sampling_rate=sr,
            text=description,
            padding=True,
            return_tensors="pt",
        )
    print("Generating...")
    # wav = model.generate(**inputs.to("cpu"), do_sample=True, guidance_scale=3, max_new_tokens=256)
    # scipy.io.wavfile.write(f"{path}/result.wav", rate=sr, data=wav[0, 0].cpu().numpy())
    print("Done.")
    
    video = VideoFileClip(str(input_video_path))
    # audio = AudioFileClip(f"{path}/result.wav")
    audio = AudioFileClip(str(f"{path}/melody.wav"))
    
    audio_duration = audio.duration

    if video.duration > audio_duration:
        video = video.subclip(0, audio_duration)
    else:
        video_clips = []
        remaining_duration = audio_duration

        while remaining_duration > 0:
            clip_duration = min(remaining_duration, video.duration)
            video_clips.append(video.subclip(0, clip_duration))
            remaining_duration -= clip_duration

        video = concatenate_videoclips(video_clips)

    video = video.set_audio(audio)

    video.write_videofile(f"{path}/video.mp4", codec='libx264', audio_codec='aac')

    return str(f"{path}/video.mp4")