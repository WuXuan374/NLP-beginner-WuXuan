# 本代码使用one-vs-all logistic regression

import csv
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
import scipy.optimize as op
import pickle
import scipy.sparse as sp

# pre-defined parameters
training_portion = 0.7
validation_portion = 0.9
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
            # 由于数据集是由phrase组成，但我只需要sentence来进行文本分类
            # 根据sentenceId, 我们只取同一个sentenceID下的第一条记录
            if int(line[1]) > i:
                sentences.append(line[2])
                labels.append(line[3])
                i = i + 1
    return sentences, labels


def read_test_data(filename):
    '''

    :param filename:
    :return: sentences: list形式
    '''
    sentences = []
    csv.register_dialect('mydialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    # 对tsv文件进行读取，把分隔符设置为\t
    with open(filename) as f:
        file_list = csv.reader(f, 'mydialect')
        i = 8544
        for line in file_list:
            # 由于数据集是由phrase组成，但我只需要sentence来进行文本分类
            # 根据sentenceId, 我们只取同一个sentenceID下的第一条记录
            if int(line[1]) > i:
                sentences.append(line[2])
                i = i + 1
    return sentences


def text_preprocessing(sentence):
    words = nltk.word_tokenize(sentence)
    words = [word for word in words if word not in string.punctuation]
    words = [word for word in words if word not in STOPWORDS and word != '']
    words = [wnl.lemmatize(word) for word in words]
    return words


def bag_of_words(sentences, WORD_INDEX, DICT_SIZE):
    '''

    :param sentences:
    :param WORD_INDEX:  ('film':0)
    :param DICT_SIZE:
    :return: bag_vector : 将sentences转化为bag_of_words的向量形式
    '''
    bag_vector = np.zeros((len(sentences), DICT_SIZE), dtype=np.int)
    i = 0
    for sentence in sentences:
        vector = np.zeros((1, DICT_SIZE), dtype=np.int)
        for word in sentence:
            if word in WORD_INDEX.keys():  # 如果该词位于词典中
                vector[0, WORD_INDEX[word]] += 1
        bag_vector[i, :] = vector
        i = i + 1
    return bag_vector


def costFunction(theta, X, Y):
    '''
    :parameter: theta : (n,)
    :return: J: cost function
    '''
    m = Y.shape[0]
    # 把theta从(n,) --> (n,1)
    theta = theta.reshape((X.shape[1], 1))
    h = sigmoid(np.dot(X, theta))

    # 没有正则化的公式
    # epsilon 为了避免divide by zero
    # J = 1 / m * (-np.dot(Y.T, np.log(h + epsilon)) - np.dot((1 - Y).T, np.log(1 - h + epsilon)))
    # grad = 1 / m * (np.dot((h - Y).T, X)).T

    # 正则化后的公式
    # 根据machine-learning-ex3给出
    lamda = 0.1
    J = 1 / m * (-np.dot(Y.T, np.log(h + epsilon)) - np.dot((1 - Y).T, np.log(1 - h + epsilon))) \
        + (lamda/(2*m)) * np.sum(np.square(theta), axis=0)
    grad = 1 / m * (np.dot((h - Y).T, X)).T + lamda / m * theta
    return J, grad


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    # axis=1 对每一行里头内容求和
    # axis = 0  对每一列里头内容求和
    return (np.exp(z.T) / (np.sum(np.exp(z), axis=1))).T


def predict(X, Y, theta_path):
    with open(theta_path, 'rb') as handle:
        theta_all = pickle.load(handle)
    # result (8544, 5)
    result = np.dot(X, theta_all)
    # 对result的每一行，取最大值所处的下标 final(8544,1)
    final = np.argmax(result, axis=1).reshape(result.shape[0], 1)
    # 计算预测标签正确的比例
    accuracy = np.sum(final == Y) / result.shape[0]
    return accuracy


def initData(sentences, WORD_INDEX, DICT_SIZE, num_of_labels):
    # 将文本使用bag_of_words表示后的结果
    # X (8544,10001)
    X = np.insert(bag_of_words(sentences, WORD_INDEX, DICT_SIZE), 0, values=1, axis=1)
    initialTheta = np.zeros(X.shape[1], dtype=float)
    finalTheta = np.zeros((X.shape[1], num_of_labels), dtype=float)
    return X, initialTheta, finalTheta


def onehot_encode(labels, number):
    y_encode = np.zeros((len(labels), 1), dtype=np.int)
    for j in range(len(labels)):
        if int(labels[j]) == number:
            y_encode[j, 0] = 1
        else:
            y_encode[j, 0] = 0
    return y_encode


if __name__ == "__main__":
    sentences, labels = read_data('movie-reviews/train.tsv')
    train_len = int(len(sentences) * training_portion)
    validation_len = int(len(sentences) * validation_portion)

    # 训练集，验证集的划分
    train_sentences = sentences[:train_len]
    train_labels = labels[:train_len]
    val_sentences = sentences[train_len:validation_len]
    val_labels = labels[train_len:validation_len]
    test_sentences = sentences[validation_len:]
    test_labels = labels[validation_len:]

    # 对文本进行预处理
    for i in range(len(train_sentences)):
        train_sentences[i] = text_preprocessing(train_sentences[i])
    for i in range(len(val_sentences)):
        val_sentences[i] = text_preprocessing(val_sentences[i])
    for i in range(len(test_sentences)):
        test_sentences[i] = text_preprocessing(test_sentences[i])

    DICT_SIZE = 5000
    freqDist = nltk.FreqDist([word for sentence in train_sentences for word in sentence])
    INDEX_WORD = dict(enumerate([word for (word, freq) in freqDist.most_common(DICT_SIZE)]))
    WORD_INDEX = dict([(value, key) for (key, value) in INDEX_WORD.items()])

    X_train, initialTheta, finalTheta = initData(train_sentences, WORD_INDEX, DICT_SIZE, num_of_labels)
    theta_path = 'logistic_theta/theta_reg_dict5000.pickle'

    # train_model
    # for i in range(num_of_labels):
    #     Y_train = onehot_encode(train_labels, i)
    #     result = op.fmin_tnc(func=costFunction, x0=initialTheta, args=(X_train, Y_train))
    #     finalTheta[:, i] = result[0]
    # with open(theta_path, 'wb') as handle:
    #     pickle.dump(finalTheta, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # predict on training data
    Y_train = np.zeros((len(train_labels), 1))
    # labels里头元素为字符，Y_train里将字符转换为数字
    for j in range(len(train_labels)):
        Y_train[j, 0] = int(train_labels[j])
    accuracy = predict(X_train, Y_train, theta_path)
    print('train_accuracy: ', accuracy)

    # predict on validation data
    Y_val = np.zeros((len(val_labels), 1))
    # labels里头元素为字符，所以创建Y,存储数字
    for j in range(len(val_labels)):
        Y_val[j, 0] = int(val_labels[j])
    X_val = bag_of_words(val_sentences, WORD_INDEX, DICT_SIZE)
    X_val = np.insert(X_val, 0, values=1, axis=1)
    accuracy = predict(X_val, Y_val, theta_path)
    print('validation accuracy: ', accuracy)

    # predict on validation data
    Y_test = np.zeros((len(test_labels), 1))
    # labels里头元素为字符，所以创建Y,存储数字
    for j in range(len(test_labels)):
        Y_test[j, 0] = int(test_labels[j])
    X_test = bag_of_words(test_sentences, WORD_INDEX, DICT_SIZE)
    X_test = np.insert(X_test, 0, values=1, axis=1)
    accuracy = predict(X_test, Y_test, theta_path)
    print('test accuracy: ', accuracy)







