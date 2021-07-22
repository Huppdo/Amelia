from playsound import playsound
from gtts import gTTS
import random

import config

def generateError(fileName):
  errorList = ["Sorry {}, I couldn't understand you", "{}, please try another time", "Well {}, this is unfortunate. I apologize for all the bugs"]
  myobj = gTTS(text=str(random.choices(errorList)).format(config.userName), lang='en-gb', slow=False)
  myobj.save("audio/"+fileName)

def generateAudio(text,fileName):
  myobj = gTTS(text=text, lang='en-gb', slow=False)
  myobj.save("audio/"+fileName)

def playAudio(fileName):
  playsound("audio/"+fileName)

def aiSpeak(text, fileName):
  if text != "__audio_error__":
    generateAudio(text,fileName)
  else:
    generateError(fileName)
  if not config.textOutput:
    playAudio(fileName)
  else:
    print("[{}] ".format(config.botName)+text)