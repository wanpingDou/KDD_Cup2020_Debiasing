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
