import numpy as np
import pickle

# read vectors from pre-trained Glove vectors and save it
words = []
idx = 0
Word_Index = {}
vectors = np.zeros((400000, 200))

with open('glove_wordvec/glove6B200d.txt', 'rb') as f:
    for line in f:
        line = line.decode().split()
        word = line[0]
        words.append(word)
        Word_Index[word] = idx
        vector = np.array(line[1:])
        vectors[idx, :] = vector
        idx = idx + 1

vectors = vectors.reshape((400000, 200)).astype('float')
print(vectors[0:4, :])

with open('glove_wordvec/words200d.pickle', 'wb') as handle:
    pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('glove_wordvec/Word_Index200d.pickle', 'wb') as handle:
    pickle.dump(Word_Index, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('glove_wordvec/wordVectors200d.npy', 'wb') as f:
    np.save(f, vectors)

