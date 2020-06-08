import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import csv
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
import pickle
import random

# pre-defined parameters
training_portion = 0.8

num_of_labels = 5
wnl = nltk.WordNetLemmatizer()
STOPWORDS = stopwords.words('english')
# 新增部分是根据所给数据得出的
STOPWORDS.extend(["'s", "'ve", '“', '”', '’', '‘', '...', '``', '--', "'ll"])
epsilon = 1e-5


def read_data(filename):
    '''

    :param filename:
    :return: sentences: list形式
    :return: labels: list形式, 元素值为0-4
    '''
    sentences = []
    labels = []
    csv.register_dialect('mydialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    # 对tsv文件进行读取，把分隔符设置为\t
    with open(filename) as f:
        file_list = csv.reader(f, 'mydialect')
        i = 0
        for line in file_list:
            # if int(line[1]) > i:
            # 只选取字符数>50的记录
            # if len(line[2]) > 50:
            #     sentences.append(line[2])
            #     labels.append(int(line[3]))
            #     i = i + 1
            sentences.append(line[2])
            labels.append(int(line[3]))

    return sentences, labels


def text_preprocessing(sentence):
    words = nltk.word_tokenize(sentence)
    words = [word for word in words if word not in string.punctuation]
    words = [word for word in words if word not in STOPWORDS and word != '']
    words = [wnl.lemmatize(word) for word in words]
    return words


def words_encode(sentences, WORD_INDEX, vocab_size):
    '''

    :param sentences:
    :param WORD_INDEX:  ('film':0)
    :param vocab_size: 每句话的长度
    :return: bag_vector : 将sentences转化为bag_of_words的向量形式
    '''
    word_vector = np.zeros((len(sentences), vocab_size), dtype=np.int)
    i = 0
    for sentence in sentences:
        j = 0
        vector = np.zeros((1, vocab_size), dtype=np.int)
        for word in sentence:
            if word in WORD_INDEX.keys():  # 如果该词位于词典中
                vector[0, j] = WORD_INDEX[word]
            else:
                # 没有在词典中出现的词，统一用0表示（词典里头1-> DICT_SIZE)
                vector[0, j] = 0
            j = j+1
        word_vector[i, :] = vector
        i = i + 1
    return word_vector


def label_encode(labels, num_of_labels):
    # 把label从0,1,2,3,4 变为[0,0,1,0,0]代表2
    label_vector = np.zeros((len(labels), num_of_labels), dtype=np.int)
    i = 0
    for label in labels:
        label_vector[i, label] = 1
        i = i+1
    return label_vector


def create_weights_matrix(DICT_SIZE, glove, INDEX_WORD, embedding_dim):
    weights_matrix = np.zeros((DICT_SIZE, embedding_dim))
    # idx = 0 指向out_of_vocabulary word
    weights_matrix[0] = np.random.normal(scale=0.6, size=(embedding_dim, ))
    for idx in INDEX_WORD.keys():
        try:
            weights_matrix[idx] = glove[INDEX_WORD[idx]]
        except KeyError:
            weights_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim, ))
    return weights_matrix


def create_embedding_layer(weights_matrix, non_trainable=False):
    (num_of_embeddings, embedding_dim) = weights_matrix.shape
    emb_layer = nn.Embedding(num_of_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


def list_shuffle(sentences,labels):
    '''
    将两个list组合起来，共同进行元素的随机排序
    原因：在原本的list里，同一类型的新闻连续出现，这对划分训练集和验证集不利
    '''
    c = list(zip(sentences, labels))
    random.shuffle(c)
    sentences, labels = zip(*c)
    return sentences, labels


sentences, labels = read_data('movie-reviews/train.tsv')
# 文本预处理
for i in range(len(sentences)):
    sentences[i] = text_preprocessing(sentences[i])
sentences, labels = list_shuffle(sentences, labels)
train_len = int(len(sentences) * training_portion)
train_sentences = sentences[:train_len]
train_labels = labels[:train_len]
val_sentences = sentences[train_len:]
val_labels = labels[train_len:]



# 使用embedding时，不再是bag_of_words, 而是每个词对应一个具体的数字0,1,2.....
# +1 的原因是， idx = 0 指向所有没在词典中出现的词
DICT_SIZE = len(set([word for sentence in train_sentences for word in sentence])) + 1
freqDist = nltk.FreqDist([word for sentence in train_sentences for word in sentence])
INDEX_WORD = dict(enumerate([word for (word, freq) in freqDist.most_common(DICT_SIZE)], start=1))
WORD_INDEX = dict([(value, key) for (key, value) in INDEX_WORD.items()])
vocab_size = 31

# load vectors
with open('glove_wordvec/words.pickle', 'rb') as handle:
    words = pickle.load(handle)
with open('glove_wordvec/Word_Index.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)
vectors = np.load('glove_wordvec/WordVectors.npy')
glove = {word: vectors[word2idx[word]] for word in words}
# 数据的读取及训练集，验证集的划分
embedding_dim = 50
target_word = set([word for sentence in train_sentences for word in sentence])
weights_matrix = create_weights_matrix(DICT_SIZE, glove, INDEX_WORD, embedding_dim)
weights_matrix = torch.tensor(weights_matrix, dtype=torch.float)
embedding_layer = create_embedding_layer(weights_matrix)



# 对文本进行编码，每一个句子[0,7,9,11]这样的形式
train_sentences = words_encode(train_sentences, WORD_INDEX, vocab_size)
val_sentences = words_encode(val_sentences, WORD_INDEX, vocab_size)
# 对标签进行编码 2->[0,0,1,0,0]
# train_labels = label_encode(train_labels, num_of_labels)
# val_labels = label_encode(val_labels, num_of_labels)


x_train = torch.tensor(train_sentences, dtype=torch.long)
y_train = torch.tensor(train_labels)
x_val = torch.tensor(val_sentences, dtype=torch.long)
y_val = torch.tensor(val_labels)


class Config(object):
    def __init__(self, savepath, DICT_SIZE, embedding_layer):
        self.savepath = savepath
        self.num_of_labels = 5
        # 论文中给出filter_size 分别为2/3/4
        self.filter_size = (2, 3, 4)
        self.dropout_rate = 0.1
        # 多出来的1是oov token
        self.vocab_size = DICT_SIZE + 1
        self.embedding_dim = 50
        self.num_of_filter = 50
        self.embedding_layer = embedding_layer


class Min_CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # num_embeddings 是文本的词典大小
        self.embedding = config.embedding_layer
        # 卷积层， 输入vocab_size = embedding_dim *1 (31*256*1)
        # 2*embedding_dim, 3*embedding_dim, 4*embedding_dim 的filter各256个
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.config.num_of_filter, kernel_size=
                      (h, self.config.embedding_dim)) for h in self.config.filter_size])
        self.dropout = nn.Dropout(self.config.dropout_rate)
        # 全连接层  参照图片，输入=filter的总数  输出-- class的数量（5）
        self.fc = nn.Linear(self.config.num_of_filter * len(self.config.filter_size), config.num_of_labels)

    def conv_and_pool(self, x, conv):
        # squeeze: dimension(A*1*B) -> (A*B)
        # parameter: dim: 哪个维度的1要删掉
        x = F.relu(conv(x)).squeeze(3)
        # 这个max_pool1d的参数不大明白 x.size(2)是channel?
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, xb):
        out = self.embedding(xb)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def getdata(train_ds, valid_ds, bs):
    return(
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=2*bs),
    )


def get_model(lr, config):
    model = Min_CNN(config)
    # weight_decay 起到regularization效果
    # opt = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-3)
    opt = optim.Adam(model.parameters(), lr=lr)
    return model, opt


def fit(epochs, opt, model, loss_func, train_dl, valid_dl, config):
    # 记录training_set, validation_set的loss,用于作图
    loss_train = []
    loss_valid = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            pred = model(xb)
            # loss_func 似乎不支持Long类型数据，所以进行类型转换
            loss = loss_func(pred.float(), yb)
            with torch.no_grad():
                epoch_loss += loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        loss_train.append(epoch_loss/len(train_dl))
        with torch.no_grad():
            val_loss = sum(loss_func(model(xb).float(), yb) for xb, yb in valid_dl)
            loss_valid.append(val_loss/(len(valid_dl)))
        print(epoch, val_loss/(len(valid_dl)))
    # 保存模型
    torch.save(model.state_dict(), config.savepath + str(epochs))
    return loss_train, loss_valid


def draw(loss_train, loss_valid, epochs):
    x = range(1, epochs+1)
    y1 = loss_train
    y2 = loss_valid
    plt.plot(x, y1, 'o-')
    plt.plot(x, y2, '.-')
    plt.title('train loss VS validation loss')
    plt.show()


def train_model(train_ds, valid_ds, lr, bs, epochs, config):
    train_dl, valid_dl = getdata(train_ds, valid_ds, bs=bs)
    model, opt = get_model(lr=lr, config=config)
    loss_func = nn.CrossEntropyLoss()

    loss_train, loss_valid = fit(epochs, opt, model, loss_func, train_dl, valid_dl, config)
    draw(loss_train, loss_valid, epochs)
    train_accu = sum(accuracy(model(xb).float(), yb) for xb, yb in train_dl)
    print(train_accu / len(train_dl))
    valid_accu = sum(accuracy(model(xb).float(), yb) for xb, yb in valid_dl)
    print(valid_accu / len(valid_dl))
    return train_accu / len(train_dl), valid_accu / len(valid_dl)


def test(train_ds, valid_ds, bs, model_path, config):
    train_dl, valid_dl = getdata(train_ds, valid_ds, bs=bs)
    model = Min_CNN(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    train_accu = sum(accuracy(model(xb).float(), yb) for xb, yb in train_dl)
    print(train_accu / len(train_dl))
    valid_accu = sum(accuracy(model(xb).float(), yb) for xb, yb in valid_dl)
    print(valid_accu / len(valid_dl))
    return train_accu / len(train_dl), valid_accu / len(valid_dl)


train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_val, y_val)
train_accu, valid_accu = train_model(train_ds, valid_ds, bs=800, lr=5e-4, epochs=10,
                                     config=Config(savepath='CNN-models/model-CNN-Glove', DICT_SIZE=DICT_SIZE,
                                                   embedding_layer=embedding_layer))
# train_accu, valid_accu = test(train_ds, valid_ds, bs=800, model_path='model-CNN-Adam_lr1e-415',
#                               config=Config(savepath='model-CNN-Adam_lr1e-415', DICT_SIZE=DICT_SIZE))

