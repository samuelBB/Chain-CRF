import numpy as np


def read_ocr(file='datasets/letter.data'):
    """
    6876 words of lengths 3-14; 130 letter features; alphabet size 26
    NOTE id_chr = dict(zip(range(26), 'abcdefghijklmnopqrstuvwxyz'))
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    chr_id = dict(zip(alphabet, range(26)))
    with open(file) as f:
        current, curr_letters, curr_word = None, [], []
        for line in f:
            l = line.split()[1:]
            letter, word, fold = chr_id[l.pop(0)], int(l.pop(1)), int(l.pop(2))
            if current != word:
                current = word
                if curr_word or curr_letters:
                    yield np.array(curr_word), np.array(curr_letters)
                curr_word, curr_letters = [], []
            curr_letters.append(letter)
            curr_word.append(np.array(map(float, l)))


def synthetic(N=50, n_feats=3, seq_len_range=(2, 4), n_labels=2):
    l, h = seq_len_range
    for _ in xrange(N):
        seq_len = np.random.randint(l, h+1)
        yield np.array([np.array([np.random.uniform()
            for _ in xrange(n_feats)]) for _ in xrange(seq_len)]), \
              np.array([np.random.randint(0, n_labels) for _ in xrange(seq_len)])