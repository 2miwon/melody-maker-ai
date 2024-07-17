from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

model = MusicGen.get_pretrained('medium', device='cuda')

def combine_midi(video_length, input_video_path, path, discription):
    model.set_generation_params(duration=video_length)
    wav = model.generate(discription, progress=True)

    for _, one_wav in enumerate(wav):
        audio_write(f'{path}/result', one_wav.cpu(), model.sample_rate, strategy="loudness")

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

    video.write_videofile(f"{path}/result.mp4", codec='libx264', audio_codec='aac')