"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx
import os
import cv2
import torch
import clip
from PIL import Image
import numpy as np
import spacy
import scipy
# from sklearn.metrics.pairwise import cosine_similarity
import av
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pprint
import json
import torchaudio
import pretty_midi
# from audiocraft.models import MusicGen
# from audiocraft.data.audio import audio_write
from pychord import Chord
from IPython.display import Audio # to display wav audio file
import logging
import asyncio
from .config import *
from pathlib import Path
from typing import List

output_folder = "output_frames"

async def extract_frames(video_path, output_folder, frame_interval=2):
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
    video_url = ""
    is_uploading: bool = False
    video_processing = False
    video_made = False
    output_video: str = ""
    video: str = ""

    # @rx.var
    # def files(self) -> list[str]:
    #     """Get the string representation of the uploaded files."""
    #     return [
    #         "/".join(p.parts[1:])
    #         for p in Path(rx.get_upload_dir()).rglob("*")
    #         if p.is_file()
    #     ]

    # def handle_upload_progress(self, progress: dict):
    #     self.uploading = True
    #     self.progress = round(progress["progress"] * 100)
    #     if self.progress >= 100:
    #         self.uploading = False

    # def cancel_upload(self):
    #     self.uploading = False
    #     return rx.cancel_upload("upload3")    

    def on_upload_progress(self, prog: dict):
        print("Got progress", prog)
        if prog["progress"] < 1:
            self.is_uploading = True
        else:
            self.is_uploading = False
        self.upload_progress = round(prog["progress"] * 100)

    async def handle_drop(self):
        try:
            State.handle_upload(rx.upload_files(upload_id="my_upload")) #, on_upload_progress=self.on_upload_progress))
        except TypeError:
            return rx.window_alert("Invalid file format")
        except Exception as ex:
            return rx.window_alert(f"Error with file upload. {ex}")

    async def handle_upload(self, files: List[rx.UploadFile]):
        self.video_processing = True
        yield
        file = files[0]
        upload_data = await file.read()
        outfile = rx.get_upload_dir() / file.filename
        
        # Save the file.
        with outfile.open("wb") as file_object:
            file_object.write(upload_data)

        # Update the img var.
        self.video = file.filename

        # self.video_processing = False

    async def get_dalle_result(self, files: list[rx.UploadFile]):
        # prompt_text: str = form_data["prompt_text"]
        self.video_made = False
        # Yield here so the image_processing take effects and the circular progress is shown.
        yield
        try:
            # response = get_openai_client().images.generate(
            #     prompt=prompt_text, n=1, size="1024x1024"
            # )
            # self.image_url = response.data[0].url
            
            file = files[0]
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename
            
            # Save the file.
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            # Update the img var.
            self.img.append(file.filename)
            
            video_path = outfile
            # get the length of the video
            cap = cv2.VideoCapture(video_path)
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Video Length: ", video_length)
            cap.release()

            if not os.path.exists(output_folder): 
                os.makedirs(output_folder)
            
            # Call the function to extract frames
            extract_frames(video_path, output_folder, frame_interval)

            """# Part 1: CLIP model emotions inference (based on the trained weights)"""

            # files = [f for f in os.listdir("output_frames")]
            # # order the files according to the frame number "frame_n"
            # # remove '.pynb_checkpoints' file
            # files = [f for f in os.listdir("output_frames") if f.endswith(".jpg") and not f.startswith(".pynb_checkpoints")]
            # files.sort()

            # print("Files: ", files)

            # # Load CLIP model (배포할 때마다만 모델 불러오기로 바꿈)
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # model, preprocess = clip.load("ViT-L/14", device)

            # # Load trained weights
            # path_to_weights = "pth/vit14_emotic_clip.pth"
            # state_dict = torch.load(path_to_weights, map_location=torch.device('cpu'))
            # model.load_state_dict(state_dict)

            # # Define your list of text classes
            # text_classes = ["Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence", "Happiness", "Pleasure", "Excitement", "Surprise", "Sympathy", "Doubt/Confusion", "Disconnection", "Fatigue", "Embarrassment", "Yearning", "Disapproval", "Aversion", "Annoyance", "Anger", "Sensitivity", "Sadness", "Disquietment", "Fear", "Pain", "Suffering"]

            # emotion_predictions = []

            # # Prepare input imag
            # for i in range(len(files)):
            #     image = preprocess(Image.open(f"output_frames/"+files[i])).unsqueeze(0).to(device)

            #     # Tokenize and move the clothing item names to the appropriate device
            #     text = torch.cat([clip.tokenize(f"The image evokes the emotion of {c}.") for c in text_classes]).to(device)

            #     # Perform inference
            #     with torch.no_grad():
            #         # Encode image and text
            #         image_features = model.encode_image(image)
            #         text_features = model.encode_text(text)

            #         # Calculate similarity scores between image and text
            #         logits_per_image, logits_per_text = model(image, text)
            #         probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            #     # Normalize image and text features
            #     image_features /= image_features.norm(dim=-1, keepdim=True)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)

            #     # Calculate similarity scores
            #     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            #     values, indices = similarity[0].topk(7)

            #     # Print the top predictions
            #     print("\nTop predictions:\n")
            #     for value, index in zip(values, indices):
            #         print(f"{text_classes[index]:>16s}: {100 * value.item():.2f}%")

            #     # store the top 2 predictions in the list
            #     top_predictions = [text_classes[index] for value, index in zip(values, indices)]
            #     top_predictions = top_predictions[:3]

            #     emotion_predictions += top_predictions

            # # print the top predictions
            # print("Predicted emotions: ", emotion_predictions)

            # # Load pre-trained word embedding model
            # nlp = spacy.load("en_core_web_md")

            # # Your list of emotions
            # final_pred = emotion_predictions # Your list of emotions goes here

            # # Convert emotions to vectors
            # emotion_vectors = np.array([nlp(emotion).vector for emotion in final_pred])

            # # Compute the average vector
            # average_vector = np.mean(emotion_vectors, axis=0)

            # # Calculate cosine similarity between the average vector and each emotion vector
            # similarities = cosine_similarity([average_vector], emotion_vectors)[0]
            # new_similarities = list(set(similarities))
            # print("Similarities: ", new_similarities)

            # #pair up the emotion_predictions and similarities in to a tuple list
            # emotion_similarities = list(zip(final_pred, similarities))
            # print("Similarities: ", emotion_similarities)

            # #sort the emotion_similarities based on the second part of each element
            # emotion_similarities.sort(key=lambda x: x[1], reverse=True)
            # print("Similarities: ", emotion_similarities)

            # final_emotions = list(dict.fromkeys(emotion_similarities))
            # print("Final emotions: ", final_emotions)

            # top_2_emotions = final_emotions[:2]
            # print("Top-2: ", top_2_emotions)
            

            # """# Part 2: Video Captioning (SpaceGPT)"""
            # # load pretrained processor, tokenizer, and model (배포할 때마다만 모델 불러오기로 바꿈)
            # image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            # tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

            # # load video
            # container = av.open(video_path)

            # # extract evenly spaced frames from video
            # seg_len = container.streams.video[0].frames
            # clip_len = model.config.encoder.num_frames
            # indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
            # frames = []
            # container.seek(0)
            # for i, frame in enumerate(container.decode(video=0)):
            #     if i in indices:
            #         frames.append(frame.to_ndarray(format="rgb24"))

            # # generate caption
            # gen_kwargs = {
            #     "min_length": 10,
            #     "max_length": 20,
            #     "num_beams": 8,
            # }
            # pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
            # tokens = model.generate(pixel_values, **gen_kwargs)
            # caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            # print("Caption: ", caption)

            # inputs=[]
            # inputs.append(final_emotions[0][0])
            # inputs.append(final_emotions[1][0])
            # inputs.append(caption)

            # print("Inputs: ", inputs)

            # """# Part 3: Extract musical features from the emotion and make prompt"""

            # pp = pprint.PrettyPrinter(indent=4)

            # # Load LLM model for RAG (배포할 때마다만 모델 불러오기로 바꿈)
            # chat = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=GROQ_API_KEY)

            # with open("emotion_library.json", 'r') as f:
            #     few_shot = json.load(f)
            
            # target_shot1 = None
            # target_shot2 = None
            # for i in range(len(few_shot)):
            #     if few_shot[i]['emotion'] == inputs[0]:
            #         target_shot1 = few_shot[i]
            #     if few_shot[i]['emotion'] == inputs[1]:
            #         target_shot2 = few_shot[i]
            # print("Target 1: ", target_shot1)
            # print("Target 2: ", target_shot2)

            # # prompt = f"{target_shot1}, {target_shot2}, give me one example chord progression with combination of the two emotions. Given genres are related to each emotion. The answer must be in 'chord = [''-''-''-''-''-''-''-'']' format."
            # prompt = f"I have these emotions: {inputs[0]}, {inputs[1]}." \
            # f"The examples of music tags related to the emotions are {target_shot1}, {target_shot2}." \
            # f"Taking consideration of such examples, I have a caption of {inputs[2]}." \
            # "I need 2 responses."\
            # "First, give me ONLY one chord progression example with combination of the two emotions. Given genres are related to each emotion. The answer must be in 'chord = [''-''-''-''-''-''-''-'']' format." \
            # "Second, Write ONLY a <Music Description Sentence> with the given music tags."\
            # "Give the response as format as written after #### tag WITHOUT ANY rationale."\
            # "####"

            # system = "You are a helpful assistant."
            # human = "{text}"
            # prompt_template = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

            # chain = prompt_template | chat
            # response = chain.invoke({"text": f"{prompt}"})
            # pp.pprint(response.__dict__)

            # response.content

            # text = response.content
            # # Find the index where '####' appears twice to separate chord and description
            # split_index = text.find('####', text.find('####') + 1)

            # # Extract chord and description
            # chord_text = text[:split_index]
            # description_text = text[split_index:]

            # # Extract the chord value
            # chord = chord_text.split('=')[1].strip().strip("['']").split('-')

            # # Remove the '####' and trim the description
            # description = [description_text.replace('####', '').strip()]

            # print("Input 1: ", inputs[0])
            # print("Input 2: ", inputs[1])
            # print("Chord: ", chord)
            # print("Description: ", description)

            # """# Part 4: Prompt MusicGen to create music"""

            self.video_processing = False
            self.video_made = True

            yield
        except Exception as ex:
            self.video_processing = False
            yield rx.window_alert(f"Error with OpenAI Execution. {ex}")


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
                    rx.cond(
                        rx.selected_files("my_upload"),
                        rx.foreach(
                            rx.selected_files("my_upload"),
                            rx.text,
                        ),
                        # rx.vstack(
                        #     rx.button(
                        #         rx.icon("refresh-cw"),
                        #         "Upload Another Video",
                        #         on_click=rx.clear_selected_files(),
                        #     ),
                        #     rx.text("Video Uploaded", font_size="1.5em"),

                        #     align="center",
                        #     spacing="2",
                        # ),
                        rx.upload(
                            rx.text(
                                "Drag and drop video or click to select video file\n max size: 10mb",
                            ),
                            accept={"video": ["video/*"]},
                            id="my_upload",
                            border="1px dotted rgb(107,99,246)",
                            padding="5em",
                            multiple=False,
                            # on_drop=State.handle_drop(),
                            on_drop=State.handle_upload(rx.upload_files(upload_id="my_upload")),
                            disabled=False,
                            max_size=10000000, # 10mb
                        ),
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
                        "Download Video with Music",
                        # type="submit",
                        size="3",
                        disabled=~State.video_made,
                        on_click=rx.download(url="/result.mp4", filename="result.mp4"),
                    ),
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