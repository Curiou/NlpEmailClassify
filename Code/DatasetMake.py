import pandas as pd, numpy as np
import re

from astropy.io.misc.tests.test_pandas import pandas
from nltk.corpus import stopwords
'''
若有错，则运行时按照错误提示下载stopwords即可
'''

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
# 导入标签对应关系，训练集和测试集的目录，h5的保存路径
from Code.CONFIG import DATASET_ONE_LABEL_DICT, TRAIN_ROOT_DIR, TEST_ROOT_DIR, TRAIN_H5_PATH, TEST_H5_PATH
from Code.CONFIG import NUM_WORDS,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM

#singleFilePath = r'C:\Users\Fisheep\Desktop\Code\py\Email\Dataset\alt.atheism\53277'
def processSingleFile(singleFilePath):
    # 参数error_bad_lines：跳过文件内出错的行
    singleDataFrame= pd.read_table(singleFilePath, header= None, encoding= 'utf8', engine= 'python', error_bad_lines=False)

    singleDataFrame.columns = ['content']
    # print(singleDataFrame)
    # print('=====================')

    def pre_clean_text(origin_text):
        # 去掉标点符号和非法字符
        origin_text = str(origin_text)
        text = re.sub('[^a-zA-Z]', ' ', origin_text)
        # 将字符全部转化为小写，并通过空格符进行分词处理,split()默认空格分开
        words = text.lower().split()
        # 去停用词
        stop_words = set(stopwords.words("english"))
        meaningful_words = [w for w in words if w not in stop_words]
        # 将剩下的词还原成str类型
        # 用空格连接列表变为string
        cleaned_text = " ".join(meaningful_words)
        return cleaned_text

    #清理数据
    singleDataFrame['content'] = singleDataFrame['content'].apply(lambda x: pre_clean_text(x))
    #压缩空维度
    valueList = np.squeeze(singleDataFrame.values)
    #print(valueList)
    #valueList = [valueList.pop(i) for i in range(len(valueList) - 1) if valueList[i] == '']

    #print('++++++++')
    # print(valueList)
    # print(' '.join(np.squeeze(singleDataFrame.values)))
    #print(valueList)
    validSentence = ' '.join(valueList)
    # print('end---------------------------')
    # print(validSentence)
    return validSentence

def makeDataFrame(train = True):
    '''

    :param train: 构建训练集还是测试集，默认true代表训练集
    :return: void 无返回值，直接将全部数据的DataFrame持久化为h5文件，
    '''
    DataFrame = pd.DataFrame(columns=['content', 'label'])
    # 根据操作的是train还是test更改目录
    if train:
        print('=================train==================')
        rootDir = TRAIN_ROOT_DIR
        hdf5SavePath = TRAIN_H5_PATH
    else:
        print('=================test==================')
        rootDir = TEST_ROOT_DIR
        hdf5SavePath = TEST_H5_PATH
    #os.chdir(rootDir)
    # 获取文件下的子目录
    subDirs = os.listdir(rootDir)
    # 记录对应异常个数
    j, k, h = 0, 0, 0
    for subDir in subDirs:
        # 转换标签
        label = DATASET_ONE_LABEL_DICT[subDir]
        for file in os.listdir(os.path.join(rootDir, subDir)):
            # 构造文件路径
            singleFilePath = os.path.join(rootDir, subDir, file)
            print('Now processing-----',singleFilePath)
            # 捕获异常，异常过多可添加处理，我这里仅是记录文件处理异常的文件个数
            try:
                # 对每个文件进行处理，返回的该文件内容有效字符构成的字符串
                sentence = processSingleFile(singleFilePath= singleFilePath)
            except pandas.errors.ParserError as e:
                j += 1
                continue
            except UnicodeDecodeError as e1:
                k +=1
                continue
            except ValueError as e2:
                h += 1
                continue
            # 添加至DataFrame里
            DataFrame = DataFrame.append(pd.DataFrame({'content': [sentence], 'label':[label]}), ignore_index= True)
    # 显示因异常未读取的文件个数
    print('pandas.errors.ParserError missed--->',j, '  UnicodeDecodeError missed--->',k, '   ValueError--->',h)
    # 打乱顺序并以覆盖的方式存储成h5文件
    DataFrame = DataFrame.sample(frac=1).reset_index(drop=True)
    DataFrame.to_hdf(hdf5SavePath, key= 'df', mode= 'w')

    print(DataFrame.head(5))
    print(DataFrame['label'].unique())
    # 统计标签种类
    print(DataFrame['label'].nunique())

def wordEmbeding(train= True):
    '''
    向量化文本，将文本转化为向量序列
    Tokenizer参数用法及含义见：
    https://tensorflow.google.cn/api_docs/python/tf/keras/preprocessing/text/Tokenizer

    :param train: 对训练集还是测试集操作，默认训练集
    :return:制作好的序列数据和对应标签np数组
    '''
    tokenizer = Tokenizer(num_words=NUM_WORDS)
    DataFrame = pd.read_hdf(TRAIN_H5_PATH)
    # 训练
    # 这里一定是用训练集的来训练
    tokenizer.fit_on_texts(DataFrame.content)
    # 如果是测试集的话还需要读测试h5
    if not train:
        DataFrame = pd.read_hdf(TEST_H5_PATH)

    # 将文本列表转换为序列，一个文本对应一个序列
    sequence = tokenizer.texts_to_sequences(DataFrame.content)
    # word_idnex = tokenizer.word_index
    # data = tokenizer.sequences_to_matrix(sequence)
    # data = tokenizer.texts_to_matrix(DataFrame.content)
    # 为序列进行填充，以0填充
    data = pad_sequences(sequences= sequence, maxlen= MAX_SEQUENCE_LENGTH)
    print(data)
    print('================')
    label = DataFrame['label'].values
    print('data.shape--->', data.shape, '   data type--->',type(data))
    print('label shape--->', label.shape, '  label type--->',type(label))
    return data, label

if __name__ =='__main__':
    # makeDataFrame(train= False)   #测试集
    # makeDataFrame()   #训练集
    # wordEmbeding(train= False)    #测试集
    wordEmbeding()  #训练集