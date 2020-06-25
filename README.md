# KDD_Cup2020_Debiasing
- KDD淘宝长尾推荐官网介绍见：https://tianchi.aliyun.com/competition/entrance/231785/information
- 比赛数据见：https://tianchi.aliyun.com/competition/entrance/231785/information
- 将数据下载到data对应的文件夹下，解压密码见：https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.3.6c3f2d3dKSeXh5&postId=102089


# 问题
- 内存优化
- 多进程加速


# 过程
- 本次比赛用户点击重复少，用户点击项目少，空间扩得比较散，用双塔DSSM、word2vec以及glove等召回效果比较差
- 主要采用itemCF召回，lightgbm排序 

# 目录

- code
	- txt_img_cosine_similarity.ipynb：计算underexpose_train/underexpose_item_feat.csv中给出的文本和图像的item间的余弦相似度，排序模型中特征备用
        - process.py                     ：数据加载，数据处理，数据embedding，数据内存优化,item聚类以及user聚类,增加session划分,用户/项目/交互特征生成
        - recall.py                      ：icf、ucf等召回
        - model.py                       ：lgbm排序模型
        - metric.py                      ：召回结果评价
        - main_V2.ipynb                  ：主函数


- data
        - process                        ：过程数据存储
        - underexpose_test               ：官方test数据
        - underexpose_train              ：官方train数据

- result
        - submit_traceb_v1.csv           ：提交数据存储 


