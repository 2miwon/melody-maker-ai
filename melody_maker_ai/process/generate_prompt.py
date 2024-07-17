from ..config import *
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

chat = ChatGroq(
    temperature=0.5, 
    model_name="llama3-70b-8192", 
    api_key=os.getenv('GROQ_API_KEY')
    )
system = "You are a helpful assistant."
human = "{text}"
prompt_template = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
with open("emotion_library.json", 'r') as f:
        few_shot = json.load(f)

def generate_prompt(inputs):
    target_shot1 = None
    target_shot2 = None
    for i in range(len(few_shot)):
      if few_shot[i]['emotion'] == inputs[0]:
        target_shot1 = few_shot[i]
      if few_shot[i]['emotion'] == inputs[1]:
        target_shot2 = few_shot[i]
    prompt = f"I have these emotions: {inputs[0]}, {inputs[1]}." \
            f"The examples of music tags related to the emotions are {target_shot1}, {target_shot2}." \
            f"Taking consideration of such examples, I have a caption of {inputs[2]}." \
            "I need 2 responses."\
            "First, generate ONLY one 8 long chord progression example with combination of the two emotions. Given genres are related to each emotion. The answer must be in 'chord = [''-''-''-''-''-''-''-'']' format." \
            "Second, Write ONLY a <Music Description Sentence> with suitable music genre, instrument, tempo, and the given emotions and caption."\
            "Give the response as format as written after #### tag WITHOUT ANY rationale."\
            "####"
    chain = prompt_template | chat
    response = chain.invoke({"text": f"{prompt}"})
    text = response.content
    split_index = text.find('####', text.find('####') + 1)

    chord_text = text[:split_index]
    description_text = text[split_index:]
    print("DS", chord_text)
    print(":", description_text)
    return chord_text, description_text
