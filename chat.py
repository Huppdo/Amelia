import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# things we need for Tensorflow
import numpy as np
import pandas as pd
import random
import json
import requests
from datetime import datetime
import spacy
import re
import twitter

import trainmodel
import processIntents
import config
import voice
import listen

nlp = spacy.load("en_core_web_sm")
twitterAPI = twitter.Api(consumer_key=config.twitterAPIKey,
                  consumer_secret=config.twitterAPISecret,
                  access_token_key=config.twitterAccessToken,
                  access_token_secret=config.twitterAccessSecret)

def clean_up_sentence(sentence):
  # tokenize the pattern - split words into array
  sentence_words = nltk.word_tokenize(sentence)
  # stem each word - create short form for word
  sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
  # tokenize the pattern
  sentence_words = clean_up_sentence(sentence)
  # bag of words - matrix of N words, vocabulary matrix
  bag = [0] * len(words)
  for s in sentence_words:
    for i, w in enumerate(words):
      if w == s:
        # assign 1 if current word is in the vocabulary position
        bag[i] = 1
        if show_details:
          print("found in bag: %s" % w)
  return np.array(bag)

def classify_local(sentence):
  global model
  ERROR_THRESHOLD = 0.25

  # generate probabilities from the model
  input_data = pd.DataFrame([bow(sentence, processIntents.words, False)], dtype=float, index=['input'])
  try:
    results = model.predict([input_data])[0]
  except:
    now = datetime.now()
    dt_string = now.strftime("error_%d%m%Y%H%M%S")
    voice.aiSpeak("There was conflicting information from the program load. I am retraining now", dt_string + ".mp3")
    processIntents.wordProcessing()
    model = trainmodel.trainModel(True)
    results = model.predict([input_data])[0]
  # filter out predictions below a threshold, and provide intent index
  results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
  # sort by strength of probability
  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in results:
    return_list.append((processIntents.classes[r[0]], str(r[1])))
  # return tuple of intent and probability
  return return_list

with open("intents.json") as readFile:
  intents = json.load(readFile)

def chat():
  global model
  global intents
  print("Begin conversation with {}: (type quit to stop)".format(config.botName))
  while True:
    if not config.textEntry:
      inp = listen.getCommand()
    else:
      inp = input("You: ")
    if inp.lower() == "quit":
      break

    result = classify_local(inp)
    #print("Message type: {}, % confidence: {}".format(result[0][0], result[0][1]))
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y%H%M%S")
    if float(result[0][1]) < float(config.minConfidence):
      print("[Critical]  {} - {}".format(inp, result))
      file1 = open("errorcategories.txt", "a+")  # append mode
      file1.write("[Critical] input: {} - generated result: {} \n".format(inp, result))
      file1.close()
      file1 = open("logs.txt", "a+")  # append mode
      file1.write("[Critical] input: {} - generated result: {} \n".format(inp, result))
      file1.close()
      voice.aiSpeak("Sorry, I cannot handle that request right now", dt_string+".mp3")
      continue
    else:
      file1 = open("logs.txt", "a+")  # append mode
      file1.write("[Info] input: {} - generated result: {} \n".format(inp, result))
      file1.close()
    for intent in intents['intents']:
      now = datetime.now()
      dt_string = now.strftime("%d%m%Y%H%M%S")
      if result[0][0] == intent["tag"]:
        if intent["type"] == "direct":
          if intent["formatBlank"] == "botName":
            formatStr = config.botName
          elif intent["formatBlank"] == "userName":
            formatStr = config.userName
          elif intent["formatBlank"] == "humantime":
            formatStr = now.strftime("%H %M")
            formatList = formatStr.split(" ")
            formatList[0] = str(int(formatList[0])%12)
            if '00' in formatList[1]:
              formatList[1] = "o clock"
            elif formatList[1][0] == '0':
              formatList[1] = str('o'+formatList[1][1])
            if formatList[0] == '0' or formatList[0] == '00':
              formatList[0] = '12'
            formatStr = formatList[0] + " " + formatList[1]
          else:
            formatStr = ""
          voice.aiSpeak(str(random.choices(intent["responses"])[0]).format(formatStr),dt_string+".mp3")
        if intent["type"] == "function":
          if result[0][0] == "intro.meetnewpeople":
            text = ""
            inpPro = nlp(inp)
            newFriendList = []
            oldFriendsList = []

            #Processing old and new friends
            with open("memoryconstruct.json","r") as readJson:
              memory = json.load(readJson)
            currentFriendsList = memory["friends"]
            for ent in inpPro.ents:
              if ent.text not in currentFriendsList:
                newFriendList.append(ent.text)
                memory["friends"].append(ent.text)
              else:
                oldFriendsList.append(ent.text)
              text += ent.text
              text += " and "

            #Speaking friends into existence
            if text == "":
              voice.aiSpeak("Who did you just say?",dt_string+".mp3")
              if not config.textEntry:
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y%H%M%S")
                inp = listen.getCommand()
              else:
                inp = input("You: ")
              inp.replace("my ","{}'s ".format(config.userName))
              if ' and ' in inp:
                inpList = inp.split(' and ')
              else:
                inpList = [str(inp)]
              for friend in inpList:
                memory["friends"].append(friend)
              voice.aiSpeak("Nice to meet you, {}".format(inp), dt_string + ".mp3")
            else:
              if not newFriendList:
                voice.aiSpeak("I've already met them, silly!", dt_string + ".mp3")
              else:
                text = text[:-5]
                voice.aiSpeak("Nice to meet you, {}".format(text),dt_string+".mp3")
            with open('memoryconstruct.json', 'w') as outfile:
              json.dump(memory, outfile, indent=2)
          if result[0][0] == "music.currentplaying":
            r = requests.get("https://spotify.domhupp.space/api/nowPlaying")
            if r.json()["song"] == "No Track Playing":
              voice.aiSpeak("Your music is either off or paused",dt_string+".mp3")
            if r.json()["song"] == "Service Currently Disabled":
              voice.aiSpeak("Your music is currently off",dt_string+".mp3")
            voice.aiSpeak("You are currently listening to {} by {}".format(r.json()["song"], r.json()["artist"]),dt_string+".mp3")
          if result[0][0] == "music.newsong":
            if inp.lower() == "play":
              r = requests.get("https://spotify.domhupp.space/api/togglePlay")
              voice.aiSpeak("Your Spotify client has began playing", dt_string + ".mp3")
            text = "undefined"
            inpPro = nlp(inp)
            for ent in inpPro.ents:
              print(ent.label_)
              text = ent.text
            if text == "undefined":
              mutated = str(inp).split("song")
              if len(mutated) > 1:
                text = mutated[1]
            r = requests.get("https://spotify.domhupp.space/api/playSong?title={}".format(text))
            r = requests.get("https://spotify.domhupp.space/api/nextSong")
            voice.aiSpeak("Now attempting to play {} on your Spotify account".format(text),dt_string+".mp3")
          if result[0][0] == "music.pause":
            r = requests.get(f"https://spotify.domhupp.space/api/togglePause&auth={config.spotifyPersonalAuth}")
            voice.aiSpeak("Your Spotify client has been paused", dt_string + ".mp3")
          if result[0][0] == "music.play": #TODO: Fix play/newsong differentiation (or combine)
            r = requests.get(f"https://spotify.domhupp.space/api/togglePlay&auth={config.spotifyPersonalAuth}")
            voice.aiSpeak("Your Spotify client has began playing", dt_string + ".mp3")
          if result[0][0] == "music.volume":
            newVolume = re.sub(r'[a-zA-Z]', r'', inp)
            newVolume = newVolume.replace(" ","").replace("%","")
            r = requests.get(f"https://spotify.domhupp.space/api/songVolume?volume={newVolume}&auth={config.spotifyPersonalAuth}")
            voice.aiSpeak("Your Spotify volume has been set to {} percent".format(str(newVolume)), dt_string + ".mp3")
          if result[0][0] == "weather.now":
            r = requests.get("https://api.openweathermap.org/data/2.5/weather?zip={}&appid=5102eb01d7b022f52538c75ccb925dfd&units=imperial".format(config.currentZip))
            weatherJson = r.json()
            if abs(weatherJson["main"]["temp"] - weatherJson["main"]["feels_like"]) > 5:
              voice.aiSpeak("The weather in {} is currently {} degrees and {}, but it feels like {} degrees".format(weatherJson["name"],weatherJson["main"]["temp"],weatherJson["weather"][0]["description"],weatherJson["main"]["feels_like"]),dt_string+".mp3")
            else:
              voice.aiSpeak("The weather in {} is currently {} degrees and {}".format(weatherJson["name"],weatherJson["main"]["temp"],weatherJson["weather"][0]["description"]),dt_string+".mp3")
          if result[0][0] == "weather.sunset":
            r = requests.get(
              "https://api.openweathermap.org/data/2.5/weather?zip={}&appid=5102eb01d7b022f52538c75ccb925dfd&units=imperial".format(
                config.currentZip))
            weatherJson = r.json()
            voice.aiSpeak("Well, weather.sunset hasn't been implemented yet.",dt_string+".mp3")
          if result[0][0] == "twitter.sendtweet":
            voice.aiSpeak("What would you like to tweet out?",dt_string + ".mp3")
            now = datetime.now()
            dt_string = now.strftime("%d%m%Y%H%M%S")
            if not config.textEntry:
              tweet = listen.getCommand()
            else:
              tweet = input("You: ")
            if len(tweet) < 200:
              twitterAPI.PostUpdate(tweet)
              voice.aiSpeak("Your tweet has been posted!", dt_string + ".mp3")
            else:
              voice.aiSpeak("That tweet is too long. Please ask me to tweet again", dt_string + ".mp3")
          if result[0][0] == "twitter.getfeed":
            voice.aiSpeak("Here are the three most recent tweets from your timeline", dt_string + ".mp3")
            lastThree = twitterAPI.GetHomeTimeline(count=3)
            for item in lastThree:
              now = datetime.now()
              dt_string = now.strftime("%d%m%Y%H%M%S")
              voice.aiSpeak(item.text, dt_string + ".mp3")
          if result[0][0] == "bot.retrain":
            voice.aiSpeak("One moment, I am retraining my model", dt_string + ".mp3")
            processIntents.wordProcessing()
            model = trainmodel.trainModel(True)
            now = datetime.now()
            dt_string = now.strftime("%d%m%Y%H%M%S")
            voice.aiSpeak("I have finished retraining my model", dt_string + ".mp3")
            with open("intents.json") as readFile:
              intents = json.load(readFile)
          if result[0][0] == "bot.silent":
            config.textOutput = True
            config.textEntry = True
            voice.aiSpeak("I have now entered text-only mode",dt_string+".mp3")
          if result[0][0] == "bot.speak":
            config.textOutput = False
            config.textEntry = False
            voice.aiSpeak("I have now entered voice mode", dt_string + ".mp3")
        if intent["type"] == "directAndLog":
          category = str(intent['tag']).split(".")
          if category[0] == 'feelings':
            with open("memoryconstruct.json","r") as readJson:
              memory = json.load(readJson)
            memory['lastMood'] = category[1]
            with open('memoryconstruct.json', 'w') as outfile:
              json.dump(memory, outfile, indent=2)
          formatStr = ""
          voice.aiSpeak(str(random.choices(intent["responses"])[0]).format(formatStr), dt_string + ".mp3")

processIntents.wordProcessing()
model = trainmodel.trainModel(False)
chat()