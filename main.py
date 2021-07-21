# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import io
import sys

import spacy
from pprint import pprint

from spacy import Language
import os.path
import json
from box import Box
import numpy as np
from spacy.tokens import Doc


class Head(object):
    def __init__(self, lines, fd=sys.stdout):
        self.lines = lines
        self.fd = fd

    def write(self, msg):
        if self.lines <= 0:
            return
        n = msg.count('\n')
        if n < self.lines:
            self.lines -= n
            return self.fd.write(msg)
        ix = 0
        while self.lines > 0:
            iy = msg.find('\n', ix + 1)
            self.lines -= 1
            ix = iy
        return self.fd.write(msg[:ix])


def pprint_head(to_print, length=10):
    pprint(to_print, stream=Head(length))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.

    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm")

    # Process whole documents
    text = ("When Sebastian Thrun started working on self-driving cars at "
            "Google in 2007, few people outside of the company took him "
            "seriously. “I can tell you very senior CEOs of major American "
            "car companies would shake my hand and turn away because I wasn’t "
            "worth talking to,” said Thrun, in an interview with Recode earlier "
            "this week.")
    doc = nlp(text)

    # Analyze syntax
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

    # Find named entities, phrases and concepts
    for entity in doc.ents:
        print(entity.text, entity.label_)


class TweetPreprocessor:
    """
    Schreiben Sie mittels Python eine Klasse TweetPreprocessor. Beim initialisieren soll Ihr
Programm die zur Verfügung gestellten fastText-Embeddings in einem Dictionary einlesen und
Spacy initialisieren. Hierzu darf auch die Funktion von der fastText-Seite verwendet werden12.
(Achten Sie hier und in allen weiteren Aufgaben darauf, dass Sie die Daten als UTF8 einlesen.)
    """
    nlp: Language

    def __init__(self):
        # nlp = spacy.load("en_core_web_sm")
        nlp_file = 'nlp_model'
        if os.path.isdir(nlp_file):
            nlp = Language().from_disk(path=nlp_file)
        else:
            nlp = self.build_npl()
            os.mkdir(nlp_file)
            nlp.to_disk(nlp_file)
        self.nlp = nlp

    def build_npl(self):
        nlp = spacy.blank('de')
        word_vectors = self.load_vectors('cc.de.100.500000.vec')
        pprint_head(word_vectors)
        first_vector = list(word_vectors.values())[0]
        nr_dim = len(list(first_vector))
        pprint({'nr_dim': nr_dim})
        nlp.vocab.reset_vectors(width=int(nr_dim))
        for word, vector in word_vectors.items():
            # pprint(word)
            # pprint(vector)
            nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab
        return nlp

    def load_vectors(self, fname: str):
        print('Reading', fname)
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        row = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = list(map(float, tokens[1:]))
            row += 1
            if not row % 10000:
                print('.', end='')
        print()
        return data

    def convert_tweet(self, tweet: str):
        """
        Schreiben Sie eine Funktion convert_tweet(), die einen string/tweet als Parameter übernimmt,
mit Hilfe von Spacy tokenisiert und eine Liste der Wortvektoren zurückgibt. Wörter die nicht
in fastText vorhanden sind, können ausgelassen werden.
        :param tweet:
        :return:
        """
        # process tweet
        doc: Doc = self.nlp(tweet)

        # print('tokens:')
        # for token in doc:
        #     print([token.text, token.pos_, token.dep_])
            # pprint(token.vector)
            # pprint(self.nlp[token.text])

        return [token.vector for token in doc]

        # print('entities')
        # for entity in doc.ents:
        #     print(entity.text, entity.label_)
        # return doc.vector

    @staticmethod
    def test_one_tweet():
        tp = TweetPreprocessor()
        vectors = tp.convert_tweet('Hello World. How are you?')
        # pprint(vectors)
        # pprint(np.shape(vectors))

    def convert_dataset(self):
        """
        Schreiben Sie nun eine zweite Funktion convert_dataset(), die einen Twitter Datensatz einliest
und jeden Tweet mit Hilfe der vorherigen Funktion verarbeitet. Repräsentieren Sie jeden Tweet
jeweils als max, min und avg (np.max(veclist, 0), np.average(veclist, 0), np.min(veclist, 0))
Vektor (dieser sollte nun jeweils die Dimension 100 haben) und speichern Sie diese mit in dem
json-Objekt (tweet["tweetmax"], tweet["tweetmin"], tweet["tweetavg"]). Der so verarbeitete
Datensatz sollte am Ende abgespeichert werden (siehe Abbildung 1).
        :return:
        """
        print('Processing dev tweets')
        with open('filtered_dev.json') as f:
            data = json.load(f)
            data = [Box(x) for x in data]
            i = 0
            for tweet in data:
                # pprint(tweet.text)
                veclist = self.convert_tweet(tweet.text)
                max = np.max(veclist, 0)
                avg = np.average(veclist, 0)
                min = np.min(veclist, 0)
                # pprint(Box(min=min, max=max, avg=avg))
                tweet.tweetmin = min.tolist()
                tweet.tweetavg = avg.tolist()
                tweet.tweetmax = max.tolist()

                i += 1
                if not i % 100:
                    print('.', end='')
        print()
        return data

    @staticmethod
    def aufgabe1():
        tp = TweetPreprocessor()
        tp.convert_tweet('Hello World. How are you?')
        dataset = tp.convert_dataset()
        # for entry in dataset:
        #     print(len(entry.tweetmin))
        with open('min-max-avg-dev.json', 'w') as f:
            dataset = list(map(prepare_for_json, dataset))
            json.dump(dataset, f, indent=2)


# NumPy array is not JSON serializable
def prepare_for_json(tweet: Box):
    tweet.tweetavg = tweet.tweetavg.to_list()
    tweet.tweetmin = tweet.tweetmin.to_list()
    tweet.tweetmax = tweet.tweetmax.to_list()
    normal = tweet.to_dict()
    return normal


if __name__ == '__main__':
    # print_hi('PyCharm')
    # TweetPreprocessor.test_one_tweet()
    TweetPreprocessor.aufgabe1()
