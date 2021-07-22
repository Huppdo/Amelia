import speech_recognition as sr

from attempt1 import config

def getCommand():
  # Initialize the recognizer
  r = sr.Recognizer()
  r.pause_threshold = 1

  with sr.Microphone() as source:
    print('{} is loading...'.format(config.name))
    r.adjust_for_ambient_noise(source, duration=1)
    print('{} is ready!'.format(config.name))
    audio = r.listen(source)

  try:
    command = r.recognize_google(audio)
    print('You said: ' + command + '\n')

  except sr.UnknownValueError:
    print('Your last command couldn\'t be heard')
    command = getCommand();

  return command