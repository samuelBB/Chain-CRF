import numpy as np
from scandir import scandir
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


def potts(n_labels, a=1., b=0.):
    A = np.full((n_labels, n_labels), b)
    np.fill_diagonal(A, a)
    return A[..., None] # NOTE expand


def synthetic(N=160, n_feats=1, lims=(100, 100), n_labels=8, normal=False):
    for _ in xrange(N):
        size = np.random.randint(lims[0], lims[1]+1)
        yield np.array([np.array([np.random.uniform() for _ in xrange(n_feats)]) for _
            in xrange(size)]), np.array([np.random.randint(0, n_labels) for _ in xrange(size)])


def synthetic_gaussian(N=160, n_nodes=100, n_labels=8):
    """
    N=160 examples of length n_nodes=100; n_feats = n_labels = 8 
    """
    X, Y = [], []
    for i in xrange(N):
        x, y = np.empty((n_nodes, n_labels)), np.empty((n_nodes, n_labels))
        for l in xrange(n_labels):
            y[:, l] = gaussian_filter(np.random.rand(n_nodes), 7)
        Y.append(np.argmax(y, axis=-1))
        for l in xrange(n_labels):
            x[:, l] = Y[i] == l
        X.append(x + np.random.normal(0., .4, (n_nodes, n_labels)))
    return X, Y, n_labels


def read_ocr(file='datasets/OCR/letter.data'):
    """
    6876 words of lengths 3-14; 130 letter features; alphabet size 26
    NOTE XXX only using pixel values as features; could also use l[2] = next-id and 
     l[4]=position (since first letter removed in each word, do position+1 ?)
    NOTE id_chr = dict(zip(range(26), 'abcdefghijklmnopqrstuvwxyz'))
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    chr_id = dict(zip(alphabet, range(26)))
    with open(file) as f:
        current, curr_letters, curr_word = None, [], []
        for line in f:
            l = line.split()
            letter, word = chr_id[l[1]], int(l[3])
            if current != word:
                current = word
                if curr_word or curr_letters:
                    yield np.array(curr_word), np.array(curr_letters)
                curr_word, curr_letters = [], []
            curr_letters.append(letter)
            curr_word.append(np.array(map(float, l[6:])))


def ocr_bigram_freqs():
    return np.load('datasets/OCR/bigram_freqs.npy')[..., None] # XXX expand


def read_gesture(path='datasets/BOFData', n_batches=20):
    """ 
    NOTE for tr/te/v split = 30/12/5, use test_pct=.3617, val_pct=.2941
    NOTE one .mat file has 30/16 split (all others 30/17)
    NOTE label set = [0,1,...,x] where x in [8,9,10,11,12,13]
    """
    for i, f in enumerate(scandir(path)):
        if i >= n_batches:
            return
        mat = loadmat(f.path)
        X, Y, V, label = [], [], [], -1
        for s, size in ('tr', mat['BOF_tr_K'].shape[1]), ('te', mat['BOF_te_K'].shape[1]):
            for i in range(size):
                X.append(np.hstack((mat['BOF_'+s+'_K'][0, i], mat['BOF_'+s+'_M'][0, i])))
                Y.append(np.squeeze(mat['label_'+s][0, i]))
                label = max(label, Y[-1].max())
                V.append(mat['videoId_'+s][0, i][0])
        yield X, Y, V, range(label+1), f.name
