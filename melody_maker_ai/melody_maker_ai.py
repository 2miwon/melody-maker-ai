"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx
import os
# import cv2
# import torch

def extract_frames(video_path, output_folder, frame_interval=2):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval in terms of number of frames
    frame_interval_frames = int(fps * frame_interval)

    # Counter to keep track of frames
    frame_count = 0

    # Read the first frame
    success, frame = cap.read()

    while success:
        # Write the frame to file if it's time to do so
        if frame_count % frame_interval_frames == 0:
            frame_filename = f"{output_folder}/frame_{frame_count // frame_interval_frames}.jpg"
            cv2.imwrite(frame_filename, frame)

        # Read the next frame
        success, frame = cap.read()

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    cap.release()

class State(rx.State):
    """The app state."""
    
    video_url = ""
    video_processing = False
    video_made = False

    def get_dalle_result(self, form_data: dict[str, str]):
        prompt_text: str = form_data["prompt_text"]
        self.video_made = False
        self.video_processing = True
        # Yield here so the image_processing take effects and the circular progress is shown.
        yield
        # try:
        #     response = get_openai_client().images.generate(
        #         prompt=prompt_text, n=1, size="1024x1024"
        #     )
        #     self.image_url = response.data[0].url
        #     self.video_processing = False
        #     self.video_made = True
        #     yield
        # except Exception as ex:
        #     self.video_processing = False
        #     yield rx.window_alert(f"Error with OpenAI Execution. {ex}")


def index():
    return rx.center(
        rx.vstack(
            rx.hstack(
                rx.color_mode.icon(),
                rx.color_mode.switch(),
                align="end",
            ),
            rx.heading("BGM Generator", font_size="1.8em"),
            rx.heading("Conditional Music Generation based on visual analysis of short video contents", font_size="1.0em", align="center"),
            rx.cond(
                State.video_processing,
                rx.chakra.circular_progress(is_indeterminate=True),
                rx.cond(
                    State.video_made,
                    rx.image(
                        src=State.video_url,
                    ),
                    rx.upload(
                        rx.text(
                            "Drag and drop video or click to select video file"
                        ),
                        id="my_upload",
                        border="1px dotted rgb(107,99,246)",
                        padding="5em",
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
                    rx.button(
                        rx.cond(
                            State.video_processing,
                            "Generating Music...",
                            rx.cond(
                                State.video_made,
                                "Download Video with Music",
                                "Generate Music by Video",
                            ),                            
                        ),
                        type="submit",
                        size="3",
                        disabled=State.video_processing,
                    ),
                    align="stretch",
                    spacing="2",
                ),
                width="100%",
                on_submit=State.get_dalle_result,
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
        # rx.divider(),
        # rx.text("Background Music Generator", font_size="1.5em", align="end"),
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