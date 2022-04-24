import numpy as np
import spacy

from layers import H5EmbeddingManager


def load_txt(p):
    with open(p) as f:
        return [line.strip() for line in f]


def bake_embedding(vocab):
    print('loading embedding...')
    embedding = H5EmbeddingManager("crawl-300d-2M.vec.h5")
    embedding_np = embedding.word_embedding_initialize(vocab,
                                                       dim_size=300,
                                                       oov_init="random")
    np.save('crawl-300d-2M.vec.npy', embedding_np)


def bake_lemma(nlp, vocab):
    lemma_map = {}
    for word in vocab:
        doc = nlp(word)
        lemma = doc[0].lemma_
        if lemma in vocab:
            lemma_map[word] = lemma
    with open('vocabularies/lemma_map.tsv', 'w') as f:
        for a, b in lemma_map.items():
            f.write(f'{a}\t{b}\n')


def bake_stopwords(nlp, vocab):
    stopwords = ['<unk>']
    for word in vocab[5:]:
        if all(not c.isalpha() for c in word):
            stopwords.append(word)
            continue

        doc = nlp(word)
        if doc[0].is_stop:
            stopwords.append(word)
    with open('vocabularies/stopwords.txt', 'w') as f:
        f.write('\n'.join(stopwords) + '\n')


if __name__ == '__main__':
    vocab = load_txt("vocabularies/word_vocab.txt")
    bake_embedding(vocab)

    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer.add_special_case('frosted-glass', [{
        'ORTH': 'frosted-glass'
    }])
    bake_lemma(nlp, vocab)
    bake_stopwords(nlp, vocab)
