from enum import Enum

class Emotion(Enum):
    PEACE = ("Peace", "평화로운", "😌")
    AFFECTION = ("Affection", "애정", "🥰")
    ESTEEM = ("Esteem", "존경", "🤩")
    ANTICIPATION = ("Anticipation", "기대", "😚")
    ENGAGEMENT = ("Engagement", "몰입", "🧐")
    CONFIDENCE = ("Confidence", "자신감", "😎")
    HAPPINESS = ("Happiness", "행복", "😊")
    PLEASURE = ("Pleasure", "즐거움", "😋")
    EXCITEMENT = ("Excitement", "흥분", "😆")
    SURPRISE = ("Surprise", "놀람", "😲")
    SYMPATHY = ("Sympathy", "공감", "🥺")
    DOUBT_CONFUSION = ("Doubt/Confusion", "의심/혼란", "🤔")
    DISCONNECTION = ("Disconnection", "무감각", "😶")
    FATIGUE = ("Fatigue", "피로", "😴")
    EMBARRASSMENT = ("Embarrassment", "당황", "😳")
    YEARNING = ("Yearning", "갈망", "🤑")
    DISAPPROVAL = ("Disapproval", "비난", "😒")
    AVERSION = ("Aversion", "혐오", "🤢")
    ANNOYANCE = ("Annoyance", "짜증", "😠")
    ANGER = ("Anger", "분노", "😡")
    SENSITIVITY = ("Sensitivity", "민감", "😬")
    SADNESS = ("Sadness", "슬픔", "😢")
    DISQUIETMENT = ("Disquietment", "불안", "😨")
    FEAR = ("Fear", "공포", "😱")
    PAIN = ("Pain", "고통", "😣")
    SUFFERING = ("Suffering", "괴로움", "😖")

    def __new__(cls, name, kor_name, emoji):
        member = object.__new__(cls)
        member._value_ = name
        member.kor_name = kor_name
        member.emoji = emoji
        member.percentage = 0
        return member

    def set_percentage(self, percentage):
        self.percentage = percentage

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