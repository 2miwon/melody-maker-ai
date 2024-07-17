from enum import Enum
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

#26개
class Emotion(Enum):
    PEACE = ("Peace", "평화로운", "😌", "var(--white-10)")
    AFFECTION = ("Affection", "애정", "🥰", "var(--pink-10)")
    ESTEEM = ("Esteem", "존경", "🤩", "var(--cyan-10)")
    ANTICIPATION = ("Anticipation", "기대", "😚", "var(--lime-10)")
    ENGAGEMENT = ("Engagement", "몰입", "🧐", "var(--teal-10)")
    CONFIDENCE = ("Confidence", "자신감", "😎", "var(--green-10)")
    HAPPINESS = ("Happiness", "행복", "😊", "var(--yellow-10)")
    PLEASURE = ("Pleasure", "즐거움", "😋", "var(--amber-10)")
    EXCITEMENT = ("Excitement", "흥분", "😆", "var(--crimson-10)")
    SURPRISE = ("Surprise", "놀람", "😲", "var(--mint-10)")
    SYMPATHY = ("Sympathy", "공감", "🥺", "var(--sky-10)")
    DOUBT_CONFUSION = ("Doubt/Confusion", "의심/혼란", "🤔", "var(--tomato-10)")
    DISCONNECTION = ("Disconnection", "무감각", "😶", "var(--grey-10)")
    FATIGUE = ("Fatigue", "피로", "😴", "var(--bronze-10)")
    EMBARRASSMENT = ("Embarrassment", "당황", "😳", "var(--plum-10)")
    YEARNING = ("Yearning", "갈망", "🤑", "var(--vilot-10)")
    DISAPPROVAL = ("Disapproval", "비난", "😒", "var(--iris-10)")
    AVERSION = ("Aversion", "혐오", "🤢", "var(--grass-10)")
    ANNOYANCE = ("Annoyance", "짜증", "😠", "var(--brown-10)")
    ANGER = ("Anger", "분노", "😡", "var(--red-10)")
    SENSITIVITY = ("Sensitivity", "민감", "😬", "var(--sand-10)")
    SADNESS = ("Sadness", "슬픔", "😢", "var(--blue-10)")
    DISQUIETMENT = ("Disquietment", "불안", "😨", "var(--orange-10)")
    FEAR = ("Fear", "공포", "😱", "var(--purple-10)")
    PAIN = ("Pain", "고통", "😣", "var(--indigo-10)")
    SUFFERING = ("Suffering", "괴로움", "😖", "var(--ruby-6)")

    def __new__(cls, name, kor_name, emoji, color):
        member = object.__new__(cls)
        member._value_ = name
        member.kor_name = kor_name
        member.emoji = emoji
        member.color = color
        member.percentage = 0.0
        return member

    def set_percentage(self, percentage):
        self.percentage = percentage

    def get_expression(self):
        return f"{self.emoji} {self.kor_name}"

    @staticmethod
    def set_mock_percentage():
        for emotion in Emotion:
            emotion.percentage = 100 / len(Emotion)

    def __str__(self):
        return self._value_
    
    @staticmethod
    def get_emotion():
        return [emotion for emotion in Emotion]

    @staticmethod
    def get_percentage_list():
        return [emotion.percentage for emotion in Emotion]

    @staticmethod
    def get_sorted_emotions():
        return sorted(Emotion, key=lambda x: x.percentage, reverse=True)
    
    @staticmethod
    def get_top_emotion(top=26) -> list:
        return Emotion.get_sorted_emotions()[:top]

    @staticmethod
    def print_prediction_list(top=26):
        for emotion in Emotion.get_sorted_emotions()[:top]:
            print(f"{emotion._value_:>16s}: {100 * emotion.percentage:.2f}%")

    @staticmethod
    def get_dict_list() -> list:
        result = []
        emotions = Emotion.get_top_emotion(top=3)
        total = sum(emotion.percentage for emotion in emotions)

        for emotion in emotions:
            result.append({
                "name": emotion.get_expression(),
                "value": int(100 * (emotion.percentage / total)),
                "fill": emotion.color,  
            })

        return result
    
    @staticmethod
    def is_initalized():
        return all(emotion.percentage != 0.0 for emotion in Emotion)
