import os
import clip
import torch
from PIL import Image
from ..config import Emotion, device

model, preprocess = clip.load("ViT-L/14", device)
# path_to_weights = "vit14_emotic_clip.pth"
# state_dict = torch.load(path_to_weights, map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)

def emotional_analysis(file_path):
    frames = [f for f in os.listdir(file_path) if f.endswith(".jpg") and not f.startswith(".pynb_checkpoints")]
    frames.sort()

    for _, ind in enumerate(frames):
        image = preprocess(Image.open(f"{file_path}/"+ind)).unsqueeze(0).to(device)

        text = torch.cat([clip.tokenize(f"The image evokes the emotion of {e}.") for e in Emotion]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, _ = model(image, text)
            _ = logits_per_image.softmax(dim=-1).cpu().numpy()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(7)

        sorted_emotions = Emotion.get_sorted_emotions()
        for value, index in zip(values, indices):
            sorted_emotions[index].set_percentage(value)