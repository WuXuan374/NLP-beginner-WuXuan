# Create RNN model using keras
import csv
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
from tensorflow import keras
import random
import matplotlib.pyplot as plt
import math

# pre-defined parameters
training_portion = 0.6
validation_portion = 0.8

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
        for line in file_list:
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
    :return: word_vector : 将sentences转化为[5,7,2,45,39,0,0]这样形式
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
                # 没有在词典中出现的词，统一用0表示（词典里头1-> vocab_size)
                vector[0, j] = 0
            j = j+1
        word_vector[i, :] = vector
        i = i + 1
    return word_vector


def label_encode(labels):
    # 把label变为numpy array
    label_vector = np.zeros((len(labels), 1), dtype=np.int)
    i = 0
    for label in labels:
        label_vector[i, 0] = label
        i = i+1
    return label_vector


def plot_graphs(history, string):
    '''
    绘制模型训练结果
    :param history:
    :param string: "acc" or "loss"
    :return:
    '''
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


def list_shuffle(sentences,labels):
    '''
    将两个list组合起来，共同进行元素的随机排序
    原因：在原本的list里，同一类型的新闻连续出现，这对划分训练集和验证集不利
    '''
    c = list(zip(sentences, labels))
    random.shuffle(c)
    sentences, labels = zip(*c)
    return sentences, labels


def test_predict(model, test_sentences, test_labels):
    '''
    使用训练好的模型，对测试集数据进行预测，并计算测试集数据的准确率
    :param model: 训练好的模型
    :return:
    accuracy : 测试集预测成功率
    predict_labels :新闻所预测的对应标签
    '''
    total = 0
    # predict_labels 是(x,6) 数组，因为它存储的是预测各个标签的概率
    predict_labels = model.predict(test_sentences)
    # 利用argmax函数，取概率最大的那个标签
    predict_labels = np.argmax(predict_labels, axis=1)
    # print(predict_labels.shape)
    for i in range(predict_labels.shape[0]):
        if predict_labels[i] == test_labels[i]:
            total += 1
    accuracy = total/predict_labels.shape[0]
    return accuracy, predict_labels


# 数据的读取及训练集，验证集的划分
sentences, labels = read_data('movie-reviews/train.tsv')
# 对文本进行预处理
for i in range(len(sentences)):
    sentences[i] = text_preprocessing(sentences[i])
# 使用list shuffle
sentences, labels = list_shuffle(sentences, labels)
train_len = int(len(sentences) * training_portion)
val_len = int(len(sentences) * validation_portion)
print(train_len)
print(val_len)
train_sentences = sentences[:train_len]
train_labels = labels[:train_len]
val_sentences = sentences[train_len:val_len]
val_labels = labels[train_len:val_len]
test_sentences = sentences[val_len:]
test_labels = labels[val_len:]



# 使用embedding时，不再是bag_of_words, 而是每个词对应一个具体的数字0,1,2.....
# 每个句子[5,7,13,2,0,0,....]
DICT_SIZE = len(set([word for sentence in train_sentences for word in sentence]))
print(DICT_SIZE)
freqDist = nltk.FreqDist([word for sentence in train_sentences for word in sentence])
INDEX_WORD = dict(enumerate([word for (word, freq) in freqDist.most_common(DICT_SIZE)], start=1))
WORD_INDEX = dict([(value, key) for (key, value) in INDEX_WORD.items()])
# 每个句子的长度（事先计算过了最长的句子，单词数就为31）
vocab_size = 31

# 文本使用bag-of-words形式表示
train_sentences = words_encode(train_sentences, WORD_INDEX, vocab_size)
val_sentences = words_encode(val_sentences, WORD_INDEX, vocab_size)
test_sentences = words_encode(test_sentences, WORD_INDEX, vocab_size)
# 对标签进行编码 变为numpy 数组
# test_labels不用编码
train_labels = label_encode(train_labels)
val_labels = label_encode(val_labels)


class Config(object):
    def __init__(self, vocab_size, model_path):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.embedding_dim = 128
        self.num_of_epochs = 1


config = Config(vocab_size, model_path="models/model-BidirectionLSTM_embedding128test.h5")
# 建立模型
model = keras.Sequential([
    # DICT_SIZE+1 -->词编码的最大值 and -1   good-1400   +1是因为位置0给到oov-token
    # input_length -->每个句子转换成向量后向量最大长度
    keras.layers.Embedding(DICT_SIZE + 1, config.embedding_dim, input_length=config.vocab_size),
    keras.layers.Bidirectional(keras.layers.LSTM(config.embedding_dim)),
    # Dense就是普通的神经网络层
    keras.layers.Dense(config.embedding_dim, activation='relu', kernel_regularizer='l2'),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# monitor -- 所观察的指标   patience =3 3个epoch内loss都没有改善则终止
# restore_best_weights --保存使loss最小时的weight
callback1 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


def scheduler(epoch):
    if epoch < 5:
        return 5e-4
    else:
        return 5e-4 * math.exp(0.1 * (5 - epoch))


callback2 = keras.callbacks.LearningRateScheduler(scheduler)

# verbose=2 输出时每个epoch占据一行
# history = model.fit(train_sentences, train_labels, validation_data=(val_sentences, val_labels),
#                     epochs=config.num_of_epochs, verbose=2, batch_size=64, callbacks=[callback1, callback2])
history = model.fit(train_sentences, train_labels, validation_data=(val_sentences, val_labels),
                    epochs=config.num_of_epochs, verbose=2, batch_size=64)
# 训练结果作图
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# 对测试集进行预测
accuracy, predict_labels = test_predict(model, test_sentences, test_labels)
print('test accuracy: ', accuracy)

model.save(config.model_path)
