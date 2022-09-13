import glob
# from io import open
from conllu import parse_incr

import nltk
from nltk.probability import LidstoneProbDist
from nltk.tag import hmm

CORPUS = "/Users/ar/work/ud-portuguese-gsd/documents/*.conllu"

def load_pos():
    sentences = []
    for fn in glob.glob(CORPUS):
        with open(fn, mode="r", encoding="utf-8") as stream:
            for tks in parse_incr(stream):
                sentences.append([(tk['form'],tk['upos']) for tk in tks])

    tag_set = set()
    symbols = set()

    cleaned_sentences = []
    for sentence in sentences:
        for i in range(len(sentence)):
            word, tag = sentence[i]
            word = word.lower()  # normalize ? 
            symbols.add(word)  
            tag_set.add(tag)
            sentence[i] = (word, tag)  
        cleaned_sentences += [sentence]

    return cleaned_sentences, list(tag_set), list(symbols)


labelled_sequences, tag_set, symbols = load_pos()
 
trainer = hmm.HiddenMarkovModelTrainer(tag_set, symbols)
hmm = trainer.train_supervised(labelled_sequences[20:],
                               estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins))

hmm.test(labelled_sequences[:20], verbose=True)

res = hmm.tag("O gato viu o rato fugir sem nada conseguir fazer .".split())
print(res)
