import json

import numpy as np


def load_json(p):
    with open(p) as f:
        return json.load(f)


def load(p):
    with open(p) as f:
        return [line.strip() for line in f]


lemma = dict(line.split('\t') for line in load('vocabularies/lemma_map.tsv'))
stopwords = load('vocabularies/stopwords.txt')


def tokenize(line):
    tokens = line.strip().split()
    lemmatized = [lemma.get(w, w) for w in tokens]
    filtered = [w for w in lemmatized if w not in stopwords]
    return filtered


def cosine_similarity(x, y):
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm < 1e-8 or y_norm < 1e-8:
        return -1.0
    return x.dot(y) / x_norm / y_norm


if __name__ == "__main__":
    vocab = load('vocabularies/word_vocab.txt')
    emb = np.load('crawl-300d-2M.vec.npy')

    similarities = np.ones((len(vocab), len(vocab)))
    for i in range(len(vocab)):
        for j in range(i):
            score = cosine_similarity(emb[i], emb[j])
            similarities[i, j] = score
            similarities[j, i] = score

    for difficulty in [1, 2, 3, 4]:
        experiment_tag = f'drqn-100-{difficulty}'
        checkpoint = f'checkpoints/{experiment_tag}/trajectory-train.json'
        corpus = load_json(checkpoint)

        action_vocab = set()
        for trajectory in corpus.values():
            for action in trajectory:
                action_vocab.update(tokenize(action))

        mask = np.array([[float(w in action_vocab) for w in vocab]])
        crest = (similarities * mask).max(axis=1)

        thr = 0.5 if difficulty == 1 else 0.4
        selected = [w for i, w in enumerate(vocab) if i < 5 or crest[i] >= thr]

        with open(f'vocabularies/crest-remove-{difficulty}.txt', 'w') as f:
            f.write('\n'.join(w for w in vocab if w not in selected) + '\n')
