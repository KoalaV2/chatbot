import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tensorflow
import random
import json
import tflearn
import pickle
import speech_recognition as sr
from gtts import gTTS
import playsound
from library.utils import say
trigger = "wake up"
r = sr.Recognizer()

def listen():
        with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
        try:
                return r.recognize_google(audio)
        except sr.UnknownValueError:
            return ""




with open("intents.json") as file:
    data = json.load(file)
#print(data)
try:
    with open("data.pickle", "rb") as f:
        words,labels,training,output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w  in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)
    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words,labels,training,output),f)

net = tflearn.input_data(shape=[None, len(training[0])]) # input layer
net = tflearn.fully_connected(net, 8) # 8 neurons
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #output layer
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8,show_metric=True)
    model.save("model.tflearn")
    
def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def chat():
    while 1:
        if listen() == trigger or input() == trigger:
            try:
                print("speak with the bot")
                say("Speak with the bot")
                while True:
                    with sr.Microphone() as source:
                        #inp = input("You: ")
                        #if inp.lower() == "quit":
                        #    break
                        inp_listen = r.listen(source)
                        inp = r.recognize_google(inp_listen)
                        results = model.predict([bag_of_words(inp,words)])
                        results_index = numpy.argmax(results)
                        tag = labels[results_index]

                        for tg in data["intents"]:
                            if tg['tag'] == tag:
                                responses = tg['responses']
                        print("You: "+ inp)
                        print(random.choice(responses))
                        say(random.choice(responses))
            except sr.UnknownValueError as err:
                print("Encountered an error: ", err)
chat()
