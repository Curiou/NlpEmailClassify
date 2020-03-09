import os
def getTrainLabelDict():
    '''
    打印类别与标签正的对应关系，需要手工复制填入下面DATASET_ONE_LABEL_DICT
    '''
    dic = {}
    dict_reverse = {}
    trainRootDir = TRAIN_ROOT_DIR
    for i, subDir in enumerate(os.listdir(trainRootDir)):
        print(subDir)
        dic[subDir] = i
        dict_reverse[i] = subDir

    print(dic)
    print()
    print(dict_reverse)



'''
更改下面这四个目录为自己对应的即可运行
环境TF2.0 
'''
NUM_WORDS = 20000    #只保留频率最高的前20000次，即是只处理文本集中出现频率最高的20000词
EMBEDDING_DIM = 256    #嵌入维度,即词向量维度
MAX_SEQUENCE_LENGTH = 300    #每个序列最大长度，多了截断，少了补0
BATCH_SIZE = 32     # 训练批大小
EPOCH = 5     #训练轮数
# 训练集目录
TRAIN_ROOT_DIR = r'C:\Users\Fisheep\Desktop\Code\py\ex\20news-bydate-train'
# 测试集目录
TEST_ROOT_DIR = r'C:\Users\Fisheep\Desktop\Code\py\ex\20news-bydate-test'
# 处理后的全体训练集数据持久化地址
TRAIN_H5_PATH = r'C:\Users\Fisheep\Desktop\Code\py\Email\Dataset\DataframeHdf5\Dataset1Train.h5'
# 处理后的全体测试集数据持久化地址
TEST_H5_PATH = r'C:\Users\Fisheep\Desktop\Code\py\Email\Dataset\DataframeHdf5\Dataset1Test.h5'
#

# 类别与标签正的对应关系
DATASET_ONE_LABEL_DICT = {'alt.atheism': 0,
                    'comp.graphics': 1,
                    'comp.os.ms-windows.misc': 2,
                    'comp.sys.ibm.pc.hardware': 3,
                    'comp.sys.mac.hardware': 4,
                    'comp.windows.x': 5,
                    'misc.forsale': 6,
                    'rec.autos': 7,
                    'rec.motorcycles': 8,
                    'rec.sport.baseball': 9,
                    'rec.sport.hockey': 10,
                    'sci.crypt': 11,
                    'sci.electronics': 12,
                    'sci.med': 13,
                    'sci.space': 14,
                    'soc.religion.christian': 15,
                    'talk.politics.guns': 16,
                    'talk.politics.mideast': 17,
                    'talk.politics.misc': 18,
                    'talk.religion.misc': 19}
# 类别与标签反对应关系
DATASET_ONE_LABEL_DICT_REVERSE = {0: 'alt.atheism',
                                  1: 'comp.graphics',
                                  2: 'comp.os.ms-windows.misc',
                                  3: 'comp.sys.ibm.pc.hardware',
                                  4: 'comp.sys.mac.hardware',
                                  5: 'comp.windows.x',
                                  6: 'misc.forsale',
                                  7: 'rec.autos',
                                  8: 'rec.motorcycles',
                                  9: 'rec.sport.baseball',
                                  10: 'rec.sport.hockey',
                                  11: 'sci.crypt',
                                  12: 'sci.electronics',
                                  13: 'sci.med',
                                  14: 'sci.space',
                                  15: 'soc.religion.christian',
                                  16: 'talk.politics.guns',
                                  17: 'talk.politics.mideast',
                                  18: 'talk.politics.misc',
                                  19: 'talk.religion.misc'}




if __name__ =='__main__':
    getTrainLabelDict()