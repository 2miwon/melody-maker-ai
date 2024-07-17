"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx
from .config import *
from .utils import *
from typing import List
from .process import *
from .views import *

class State(rx.State):    
    is_uploading: bool = False
    video_processing = False
    output_video: str = ""
    output_folder: str = "assets/"
    result: dict = {}
    chart_data = []
    unique_id = generate_uuid()

    def on_upload_progress(self, prog: dict):
        print("Got progress", prog)
        if prog["progress"] < 1:
            self.is_uploading = True
        else:
            self.is_uploading = False
        self.upload_progress = round(prog["progress"] * 100)

    async def handle_upload(self, files: List[rx.UploadFile]):
        self.video_processing = True
        yield
        file = files[0]
        upload_data = await file.read()
        output_fname = self.unique_id #+ get_file_extension(file.filename)
        outfile_path = rx.get_upload_dir() / output_fname
        unique_path = self.output_folder + self.unique_id

        with outfile_path.open("wb") as file_object:
            file_object.write(upload_data)

        # 1. 프레임 쪼개기
        video_length = extract_frames(outfile_path, unique_path)
        
        # 2. clip으로 감정 분석
        emotional_analysis(unique_path)

        self.chart_data = Emotion.get_dict_list()
        # Emotion.print_prediction_list()

        inputs = [emotion._value_ for emotion in Emotion.get_top_emotion(top=3)]
        
        # 3. spaceformer로 video captioning
        caption = generate_caption(outfile_path)
        
        inputs.append(caption)
        print("caption", caption)

        # 4. Llama로 2과 3의 정보 합쳐서 적절한 music prompt 생성
        chord, prompt = generate_prompt(inputs)

        print("chord: ", chord)
        print("prompt: ", prompt)

        # 5. 생성된 music prompt를 musicgen에게 주고 음악 생성
        create_midi_with_beat(chord, unique_path)

        # 6. 생성된 음악과 영상 합치기
        combine_midi(video_length, outfile_path, unique_path, prompt)
        self.output_video = unique_path + "/result.mp4"

        self.video_processing = False
    

def index():
    return rx.center(
        rx.vstack(
            rx.hstack(
                rx.color_mode.icon(),
                rx.color_mode.switch(),
                align="end",
            ),
            rx.heading("BGM Generator", font_size="1.8em"),
            rx.cond(
                State.chart_data,
                pie_chart(State.chart_data),
                rx.heading("Conditional Music Generation based on visual analysis of short video contents", font_size="1.0em", align="center"),
            ),
            rx.cond(
                State.video_processing,
                rx.chakra.circular_progress(is_indeterminate=True),
                rx.cond(
                    State.output_video,
                    # rx.text("Video with Music Generated", font_size="1.0em"),
                    rx.video(
                        url=State.output_video,
                        width="100%",
                        height="auto",
                    ),
                    rx.cond(
                        rx.selected_files("my_upload"),
                        rx.foreach(
                            rx.selected_files("my_upload"),
                            rx.text,
                        ),
                        upload_box(State),
                    ),
                ),
            ),
            rx.form(
                rx.vstack(
                    # rx.input(
                    #     id="prompt_text",
                    #     placeholder="(optional) Enter a prompt",
                    #     size="3",
                    #     disabled=State.video_processing | State.video_made,
                    # ),
                    download_button(State),
                    align="stretch",
                    spacing="2",
                ),
                width="100%",
                # on_submit=State.get_dalle_result(rx.upload_files(upload_id="my_upload")), 
            ),
            rx.divider(),
            rx.flex(
                rx.text(
                    "powered by ",
                    rx.link("Reflex", href="https://reflex.dev/"),
                    align="center",
                ),
            ),
        
            width="25em",
            # bg="white",
            padding="2em",
            align="center",
            gap="1.5em",
        ),
        align="center",
        width="100%",
        height="100vh",
        background="""radial-gradient(circle at 10% 0%, rgba(103, 58, 183, 0.3), rgba(41, 47, 69, 0) 50%),
            radial-gradient(circle at 90% 10%, rgba(33, 150, 243, 0.2), rgba(41, 47, 69, 0) 50%),
            radial-gradient(circle at 50% 50%, rgba(63, 81, 181, 0.1), rgba(41, 47, 69, 0) 70%)""",
    )


# Add state and page to the app.
app = rx.App(
    theme=rx.theme(
        appearance="inherit", has_background=True, radius="medium", accent_color="mint"
    ),
)
app.add_page(index, title="BGM Generator")