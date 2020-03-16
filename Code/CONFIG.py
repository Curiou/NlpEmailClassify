import os
def getTrainLabelDict():
    '''
    打印类别与标签正的对应关系，需要手工复制填入下面DATASET_ONE_LABEL_DICT
    '''
    dic = {}
    dict_reverse = {}
    trainRootDir = THREE_TRAIN_ROOT_PATH
    for i, subDir in enumerate(os.listdir(trainRootDir)):
        print(subDir)
        dic[subDir] = i
        dict_reverse[i] = subDir

    print(dic)
    print()
    print(dict_reverse)


'''
更改下面这四个目录为自己对应的，停用词文件路径也需要更改
环境TF2.0 
'''
NUM_WORDS = 20000    #只保留频率最高的前20000次，即是只处理文本集中出现频率最高的20000词
EMBEDDING_DIM = 256    #嵌入维度,即词向量维度
MAX_SEQUENCE_LENGTH = 300    #每个序列最大长度，多了截断，少了补0
BATCH_SIZE = 32     # 训练批大小
EPOCH = 8     #训练轮数

# 停止词文件路径
STOP_WORDS_FILE_PATH = r"C:\Users\Fisheep\Desktop\Code\py\Email\Code\Dataset3\StopWords.txt"
#########################  数据集1  ################################
# 训练集目录
TRAIN_ROOT_DIR = r'C:\Users\Fisheep\Desktop\Code\py\ex\20news-bydate-train'
# 测试集目录
TEST_ROOT_DIR = r'C:\Users\Fisheep\Desktop\Code\py\ex\20news-bydate-test'
# 处理后的全体训练集数据持久化地址
TRAIN_H5_PATH = r'C:\Users\Fisheep\Desktop\Code\py\Email\Dataset\DataframeHdf5\Dataset1Train.h5'
# 处理后的全体测试集数据持久化地址
TEST_H5_PATH = r'C:\Users\Fisheep\Desktop\Code\py\Email\Dataset\DataframeHdf5\Dataset1Test.h5'
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
#########################  数据集1  ################################

#########################  数据集2  ################################
TWO_TRAIN_ROOT_PATH = r"C:\Users\Fisheep\Desktop\Code\py\Dataset2\数据集3UTF8格式 (1)\数据集3UTF8格式\train"
TWO_TEST_ROOT_PATH = r"C:\Users\Fisheep\Desktop\Code\py\Dataset2\数据集3UTF8格式 (1)\数据集3UTF8格式\test"

TWO_TRAIN_H5_PATH = r"C:\Users\Fisheep\Desktop\Code\py\Email\Dataset\DataframeHdf5\Dataset2Train.h5"
TWO_TEST_H5_PATH = r"C:\Users\Fisheep\Desktop\Code\py\Email\Dataset\DataframeHdf5\Dataset2Test.h5"

DATASET_TWO_LABEL_DICT = {'C11-Space': 0,
                          'C15-Energy': 1,
                          'C16-Electronics': 2,
                          'C17-Communication': 3,
                          'C19-Computer': 4,
                          'C23-Mine': 5,
                          'C29-Transport': 6,
                          'C3-Art': 7,
                          'C31-Enviornment': 8,
                          'C32-Agriculture': 9,
                          'C34-Economy': 10,
                          'C35-Law': 11,
                          'C36-Medical': 12,
                          'C37-Military': 13,
                          'C38-Politics': 14,
                          'C39-Sports': 15,
                          'C4-Literature': 16,
                          'C5-Education': 17,
                          'C6-Philosophy': 18,
                          'C7-History': 19}
DATASET_TWO_LABEL_DICT_REVERSE = {0: 'C11-Space',
                                  1: 'C15-Energy',
                                  2: 'C16-Electronics',
                                  3: 'C17-Communication',
                                  4: 'C19-Computer',
                                  5: 'C23-Mine',
                                  6: 'C29-Transport',
                                  7: 'C3-Art',
                                  8: 'C31-Enviornment',
                                  9: 'C32-Agriculture',
                                  10: 'C34-Economy',
                                  11: 'C35-Law',
                                  12: 'C36-Medical',
                                  13: 'C37-Military',
                                  14: 'C38-Politics',
                                  15: 'C39-Sports',
                                  16: 'C4-Literature',
                                  17: 'C5-Education',
                                  18: 'C6-Philosophy',
                                  19: 'C7-History'}
#########################  数据集2  ################################

#########################  数据集3  ################################
THREE_TRAIN_ROOT_PATH = r"C:\Users\Fisheep\Desktop\Code\py\Dataset2\学科数据集\train"
THREE_TEST_ROOT_PATH = r"C:\Users\Fisheep\Desktop\Code\py\Dataset2\学科数据集\test"
THREE_TRAIN_H5_PATH = r"C:\Users\Fisheep\Desktop\Code\py\Email\Dataset\DataframeHdf5\Dataset3Train.h5"
THREE_TEST_H5_PATH = r"C:\Users\Fisheep\Desktop\Code\py\Email\Dataset\DataframeHdf5\Dataset3Test.h5"
DATASET_THREE_LABEL_DICT = {'sougou_allaoyun': 0,
                            'sougou_allfangchan': 1,
                            'sougou_allhulianwang': 2,
                            'sougou_alljiankang': 3,
                            'sougou_alljiaoyu': 4,
                            'sougou_alljunshi': 5,
                            'sougou_alllvyou': 6,
                            'sougou_allqiche': 7,
                            'sougou_allshangye': 8,
                            'sougou_allshishang': 9,
                            'sougou_alltiyu': 10,
                            'sougou_allwenhua': 11,
                            'sougou_allxueke': 12,
                            'sougou_allyule': 13}
DATASET_THREE_LABEL_DICT_REVERSE = {0: 'sougou_allaoyun',
                                    1: 'sougou_allfangchan',
                                    2: 'sougou_allhulianwang',
                                    3: 'sougou_alljiankang',
                                    4: 'sougou_alljiaoyu',
                                    5: 'sougou_alljunshi',
                                    6: 'sougou_alllvyou',
                                    7: 'sougou_allqiche',
                                    8: 'sougou_allshangye',
                                    9: 'sougou_allshishang',
                                    10: 'sougou_alltiyu',
                                    11: 'sougou_allwenhua',
                                    12: 'sougou_allxueke',
                                    13: 'sougou_allyule'}


#########################  数据集3  ################################
if __name__ =='__main__':
    getTrainLabelDict()