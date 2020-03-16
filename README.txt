
1、CONFIG里面更改停止词对应

2、更改CONFIG配置文件内的不同数据集的这四个目录为自己的目录：
    TRAIN_ROOT_DIR
    TEST_ROOT_DIR
    TRAIN_H5_PATH
    TEST_H5_PATH

3、DatasetMake里面运行dataset2函数和makeDataFrame制作数据集
    dataset2函数：
        用于制作数据集2和数据集3（学科数据）；
    makeDataFrame：
        仅用于制作数据集1；

4、在Network文件里面更改为对应的数据集即可运行