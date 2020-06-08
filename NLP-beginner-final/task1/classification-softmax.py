# 使用自己定义的softmax regression 来完成文本分类任务
import csv
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
import scipy.optimize as op
import pickle
import matplotlib.pyplot as plt
import random

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
            # 所有phrase都拿来训练
            sentences.append(line[2])
            labels.append(line[3])
    return sentences, labels


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
    bag_vector = np.zeros((len(sentences), DICT_SIZE), dtype=np.int8)
    i = 0
    for sentence in sentences:
        vector = np.zeros((1, DICT_SIZE), dtype=np.int8)
        for word in sentence:
            if word in WORD_INDEX.keys():  # 如果该词位于词典中
                vector[0, WORD_INDEX[word]] += 1
        bag_vector[i, :] = vector
        i = i + 1
    return bag_vector


def costFunction(theta, X, Y):
    '''
    :parameter: theta : (n,)
    :parameter: Y: one - hot vector
    :return: J: cost function
    '''
    m = Y.shape[0]
    # 把theta从(n,) --> (n/5,5)
    theta = theta.reshape((int(theta.shape[0]/num_of_labels), num_of_labels))
    h = softmax(np.dot(X, theta))
    # epsilon 为了避免divide by zero
    # After regularization
    # 根据machine-learning-ex3给出
    lamda = 0.1
    J = -1 / m * np.sum(Y * np.log(h + epsilon)) + (lamda/(2*m)) * np.sum(np.square(theta))
    grad = -1 / m * (np.dot((Y - h).T, X)).T + lamda / m * theta
    grad = grad.reshape(theta.shape[0] * num_of_labels)
    return J, grad


def fit(epochs, theta, costfunc, bs, x_train, y_train, x_val, y_val, lr):
    loss_train = []
    loss_val = []
    for epoch in range(epochs):
        print('epoch ', epoch)
        epoch_cost_train = 0
        epoch_cost_val = 0
        # 计算train loss
        for i in range((x_train.shape[0] - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            cost, grad = costfunc(theta, xb, yb)
            theta = theta - lr*grad
            epoch_cost_train += cost

        loss_train.append(epoch_cost_train/((x_train.shape[0] - 1) // bs + 1))
        print('train loss: ', epoch_cost_train/((x_train.shape[0] - 1) // bs + 1))
        # 计算valid_loss
        for i in range((x_val.shape[0] - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_val[start_i:end_i]
            yb = y_val[start_i:end_i]
            cost, grad = costfunc(theta, xb, yb)
            epoch_cost_val += cost
        loss_val.append(epoch_cost_val/((x_val.shape[0] - 1) // bs + 1))
        print('val_loss: ', epoch_cost_val / ((x_val.shape[0] - 1) // bs + 1))
    return theta, loss_train, loss_val


def softmax(z):
    # axis=1 对每一行里头内容求和、求最值  axis = 0  对每一列里头内容求和/求最值
    mymax = np.max(z, axis=1).reshape((z.shape[0], 1))
    z = np.exp(z-mymax) / (np.sum(np.exp(z-mymax), axis=1)).reshape((z.shape[0], 1))
    return z


def predict(x_test, y_test, theta_path):
    with open(theta_path, 'rb') as handle:
        theta_all = pickle.load(handle)
    theta_all = theta_all.reshape((int(theta_all.shape[0]/num_of_labels), num_of_labels))
    result = np.dot(x_test, theta_all)
    # 对result的每一行，取最大值所处的下标 final(8544,1)
    final = np.argmax(result, axis=1).reshape(result.shape[0], 1)
    # 计算预测标签正确的比例
    accuracy = np.sum(final == y_test) / result.shape[0]

    return accuracy, final


def initData(train_sentences, val_sentences, test_sentences, WORD_INDEX, DICT_SIZE, num_of_labels):
    # 将文本使用bag_of_words表示后的结果
    # 添加bias项
    x_train = np.insert(bag_of_words(train_sentences, WORD_INDEX, DICT_SIZE), 0, values=1, axis=1)
    x_train = np.array(x_train, dtype=np.int8)
    initialTheta = np.zeros(x_train.shape[1] * num_of_labels, dtype=float)
    finalTheta = np.zeros((x_train.shape[1], num_of_labels), dtype=float)
    x_val = np.insert(bag_of_words(val_sentences, WORD_INDEX, DICT_SIZE), 0, values=1, axis=1)
    x_val = np.array(x_val, dtype=np.int8)
    x_test = np.insert(bag_of_words(test_sentences, WORD_INDEX, DICT_SIZE), 0, values=1, axis=1)
    x_test = np.array(x_test, dtype=np.int8)
    return x_train, x_val, x_test, initialTheta, finalTheta


def draw(loss_train, loss_val, epochs):
    x = range(1, epochs+1)
    y1 = loss_train
    y2 = loss_val
    plt.plot(x, y1, 'o-')
    plt.plot(x, y2, '.-')
    plt.title('train loss vs validation loss ')
    plt.legend(['train loss', 'validation loss'])
    plt.show()


def one_hot_encode(labels):
    y_encode = np.zeros((len(labels), num_of_labels), dtype=np.int)
    for j in range(len(labels)):
        i = int(labels[j])
        # Y的每一行0,0,1,0,0
        y_encode[j, i] = 1
    return y_encode


def list_shuffle(sentences,labels):
    '''
    将两个list组合起来，共同进行元素的随机排序
    原因：在原本的list里，同一类型的新闻连续出现，这对划分训练集和验证集不利
    '''
    c = list(zip(sentences, labels))
    random.shuffle(c)
    sentences, labels = zip(*c)
    return sentences, labels


def calculate_accuracy(x, y, theta_path, bs):
    # 计算accuracy时同样需要mini-batch, 否则出现MemoryError
    accuracy = 0
    for i in range((x.shape[0] - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x[start_i:end_i]
        yb = y[start_i:end_i]
        batch_accuracy, _ = predict(xb, yb, theta_path)
        accuracy += batch_accuracy
    return accuracy/((x.shape[0] - 1) // bs + 1)


if __name__ == "__main__":
    # 五类新闻内容的读取
    sentences, labels = read_data('movie-reviews/train.tsv')
    # 对文本进行预处理
    for i in range(len(sentences)):
        sentences[i] = text_preprocessing(sentences[i])
    sentences, labels = list_shuffle(sentences, labels)
    train_len = int(len(sentences) * training_portion)
    validation_len = int(len(sentences) * validation_portion)

    # 训练集，验证集的划分
    train_sentences = sentences[:train_len]
    train_labels = labels[:train_len]
    val_sentences = sentences[train_len: validation_len]
    val_labels = labels[train_len: validation_len]
    test_sentences = sentences[validation_len:]
    test_labels = labels[validation_len:]

    # 词典构造
    DICT_SIZE = 10000
    freqDist = nltk.FreqDist([word for sentence in sentences for word in sentence])
    INDEX_WORD = dict(enumerate([word for (word, freq) in freqDist.most_common(DICT_SIZE)]))
    WORD_INDEX = dict([(value, key) for (key, value) in INDEX_WORD.items()])
    # 相关数据初始化
    X_train, X_val, X_test, initialTheta, finalTheta = initData(train_sentences, val_sentences, test_sentences,
                                                                WORD_INDEX, DICT_SIZE, num_of_labels)

    # one-hot vector
    Y_train = one_hot_encode(train_labels)
    Y_val = one_hot_encode(val_labels)

    # Y_test每行为0-4的数字
    Y_test = np.zeros((len(test_labels), 1), dtype=np.int)
    for i in range(len(test_labels)):
        Y_test[i, 0] = test_labels[i]
    Y_train_number = np.zeros((len(train_labels), 1), dtype=np.int)
    for i in range(len(train_labels)):
        Y_train_number[i, 0] = train_labels[i]
    Y_val_number = np.zeros((len(val_labels), 1), dtype=np.int)
    for i in range(len(val_labels)):
        Y_val_number[i, 0] = val_labels[i]
    theta_path = 'softmax_theta/dict10000_epochs30_lr0.12_bs800_shuffle.pickle'

    # train
    # epochs = 30
    # finalTheta, loss_train, loss_val = fit(epochs=epochs, theta=initialTheta, costfunc=costFunction,
    #                                        bs=800, x_train=X_train, y_train=Y_train,
    #                                        x_val=X_val, y_val=Y_val, lr=0.12)
    #
    # with open(theta_path, 'wb') as handle:
    #     pickle.dump(finalTheta, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # draw(loss_train, loss_val, epochs=epochs)


    # test
    train_accuracy = calculate_accuracy(X_train, Y_train_number, theta_path, bs=1600)
    print('train accuracy: ', train_accuracy)
    validation_accuracy = calculate_accuracy(X_val, Y_val_number, theta_path, bs=1600)
    print('validation accuracy: ', validation_accuracy)
    # 测试集数量少，不需要分批计算accuracy
    test_accuracy, predict_labels = predict(X_test, Y_test, theta_path)
    print('test accuracy:', test_accuracy)
    # 部分预测结果
    for i in range(5):
        print(test_sentences[i])
        print('predict_labels: ', predict_labels[i, 0])










