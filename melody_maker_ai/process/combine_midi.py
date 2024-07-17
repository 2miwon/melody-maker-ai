from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
import torchaudio
import scipy

processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

def combine_midi(video_length, input_video_path, path, description):
    melody, sr = torchaudio.load("output_frames/melody.wav")
    inputs = processor(
            audio=melody[0],
            sampling_rate=sr,
            text=description,
            padding=True,
            return_tensors="pt",
        )
    wav = model.generate(**inputs.to("cpu"), do_sample=True, guidance_scale=3, max_new_tokens=256)
    scipy.io.wavfile.write(f"{path}/result.wav", rate=sr, data=wav[0, 0].cpu().numpy())

    video = VideoFileClip(input_video_path)
    audio = AudioFileClip(f"{path}/result.wav")
    
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