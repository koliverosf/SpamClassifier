import string
import spacy as sp


en = sp.load('en_core_web_sm')

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def aanumbers(words):
    for i in range(len(words)):
        if has_numbers(words[i]):
            words[i] = "aanumbers"
    return words


def singletterword(words):
    words = [x for x in words if len(x) > 1]
    return words


def remove_punct(words):
    for i in range(len(words)):
        words[i] = words[i].translate(str.maketrans('', '', string.punctuation))
    return words


def exploder(words: list, vocab: list):
    answer = {elt.strip(): 0 for elt in words}
    for i in range(len(words)):
        if words[i] in vocab:
            value = answer.get(words[i]) + 1
            answer.update({words[i]: value})
    return answer

def stopwords_remover(words):
    words = [stopwords.text for stopwords in en(' '.join(words)) if not stopwords.is_stop]
    return words

def translate_target(word):
    if word =='ham':
        return 0
    elif word == 'spam':
        return 1

def translate_pred(word):
    if word == 0:
        return 'ham'
    elif word == 1:
        return 'spam'
