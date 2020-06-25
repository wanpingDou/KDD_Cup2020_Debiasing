#coding:utf-8

import pandas as pd
import numpy as np
import math
import psutil
import gensim

try:
    from glove import Glove
    from glove import Corpus
except:
    print('have not glove')
from collections import Counter
from tqdm import tqdm
from collections import defaultdict

def topk_recall_association_rules_qyxs_icf(click_all, 
                             dict_label, 
                             k=100):
    """
    修改自author: 青禹小生 鱼遇雨欲语与余
    
    click_all：点击数据
    dict_label：训练集user-item字典
    k: 召回数量，不足热补
    
    topk_recall：用户召回表
    matrix_association_rules：项目得分矩阵
        
    """
    
    group_by_col, agg_col = 'user_id', 'item_id'
 
    # data_ = click_all.groupby(['user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()
    
    data_ = click_all.groupby(['user_id'])[['item_id','time']].agg({'item_id':lambda x:','.join(list(x)), 'time':lambda x:','.join(list(x))}).reset_index()

    hot_list = list(click_all['item_id'].value_counts().index[:].values)
    stat_cnt = Counter(list(click_all['item_id'])) 
    stat_length = np.mean([ len(item_txt.split(',')) for item_txt in data_['item_id']])
  
    matrix_association_rules = {}
    print('------- association rules matrix 生成 ---------')
    for i, row in tqdm(data_.iterrows()):
        
        list_item_id = row['item_id'].split(',')
        list_time = row['time'].split(',')
        len_list_item = len(list_item_id)

        for i, (item_i, time_i) in enumerate(zip(list_item_id,list_time)):
            for j, (item_j, time_j) in enumerate(zip(list_item_id,list_time)):
                
                t = np.abs(float(time_i)-float(time_j))
                d = np.abs(i-j)
 
                if item_i not in matrix_association_rules:
                        matrix_association_rules[item_i] = {}
                if item_j not in matrix_association_rules[item_i]:
                        matrix_association_rules[item_i][item_j] = 0
                # 改为不区分正反向
                matrix_association_rules[item_i][item_j] += np.float32(1 * 1.0 * (0.95**(d-1)) / (1 + t * 10000) / np.log(1 + len_list_item))

                  
    assert len(matrix_association_rules.keys()) == len(set(click_all['item_id']))

    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- association rules 召回 ---------')
    for i, row in tqdm(data_.iterrows()):
        
        list_item_id = row['item_id'].split(',')

        dict_item_id_score = {}
        for i, item_i in enumerate(list_item_id[::-1]):
            for item_j, score_similar in sorted(matrix_association_rules[item_i].items(), key=lambda x: x[1], reverse=True)[0:k]:
                if item_j not in list_item_id:
                    if item_j not in dict_item_id_score:
                        dict_item_id_score[item_j] = 0
                    sigma = 0.25
                    dict_item_id_score[item_j] += 1.0 / (1 + sigma * i) * score_similar
                
        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
 
        # 不足的热度补全
        if len(dict_item_id_score_topk) < k:
            for i, item in enumerate(hot_list):
                if (item not in list_item_id) and (item not in dict_item_id_set):
                    item_similar = item
                    score_similar = - i - 88 
                    dict_item_id_score_topk.append( (item_similar, score_similar) )
                if len(dict_item_id_score_topk) == k:
                    break

        assert len(dict_item_id_score_topk) == k
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])
            
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')
    
    # 类别转化，节省内存
    topk_recall['user_id'] = topk_recall['user_id'].astype('str')
    topk_recall['item_similar'] = topk_recall['item_similar'].astype('str')
    topk_recall['score_similar'] = topk_recall['score_similar'].astype('float32')

    return topk_recall, matrix_association_rules





def topk_recall_association_rules_ucf(click_all, 
                           dict_label, 
                           k=100):
    """
    
    click_all：点击数据
    dict_label：训练集user-item字典
    k: 召回数量，不足热补
    
    topk_recall：用户召回表
    sim_user_corr：项目得分矩阵
        
    """

    user_item_ = click_all.groupby('user_id')['item_id'].agg(lambda x: list(x)).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['item_id']))
    # data_ =  click_all.groupby(['user_id'])['item_id'].agg(lambda x: ','.join(list(x))).reset_index()
    
    hot_list = list(click_all['item_id'].value_counts().index[:].values)
    user_cnt = defaultdict(int)
    item_users = {}
    for u, u_item in tqdm(user_item_dict.items()):
        for item in u_item:
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(u)

    c = dict()
    print('------- UCF association rules matrix 生成 ---------')
    for item, u_user in tqdm(item_users.items()):
        for u in u_user:
            user_cnt[u] += 1
            c.setdefault(u, {})
            for v in u_user:
                if v == u:
                    continue
                c[u].setdefault(v, 0)
                c[u][v] += 1/math.log(1+len(u_user))

    sim_user_corr = c.copy()
    for i, related_user in tqdm(c.items()):
        for j, cij in related_user.items():
            sim_user_corr[i][j] = cij / math.sqrt(user_cnt[i] * user_cnt[j])

    assert len(sim_user_corr.keys()) == len(set(click_all['user_id']))

    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('-------UCF association rules 召回 ---------')
    for user_id in tqdm(user_item_['user_id'].unique()):
       # num of id :18505  
        dict_item_id_score = {}
        interacted_items = user_item_dict[user_id]
    #     每个user找到相似的100个user
    #     print(len(sim_user_corr[user_id]))
    #     这里找不到一千个用户， 很多时候都是几十个相似度的
        for relate_user, score_similar in sorted(sim_user_corr[user_id].items(), key=lambda x:x[1], reverse=True)[0:1600]:
            # 类似user点击过的item倒排
            
            for i, item in enumerate(user_item_dict[relate_user][::-1]):
                
                if item not in interacted_items:
                    dict_item_id_score.setdefault(item, 0)
    #                 dict_item_id_score[item] += score_similar
                    sigma = 0.003
                    dict_item_id_score[item] +=  1.0 / (1 + sigma * i) * score_similar
                    
    #     print(len(dict_item_id_score))
        
        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda d: d[1], reverse=True)[:k]

        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        
        #不足的热度补全
        if len(dict_item_id_score_topk) < k:
            for i, item in enumerate(hot_list):
                if (item not in interacted_items) and (item not in dict_item_id_set):
                    item_similar = item
                    score_similar = - i - 100 
                    dict_item_id_score_topk.append( (item_similar, score_similar) )  
                if len(dict_item_id_score_topk) == k:
                    break
    

        assert len(dict_item_id_score_topk) == k
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(user_id)


    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    # 类别转化，节省内存
    topk_recall['user_id'] = topk_recall['user_id'].astype('str')
    topk_recall['item_similar'] = topk_recall['item_similar'].astype('str')
    topk_recall['score_similar'] = topk_recall['score_similar'].astype('float32')

    return topk_recall, sim_user_corr




def topk_recall_association_rules_icf(click_all, 
                          dict_label, 
                          k=100):
    """
        关联矩阵：按距离加权 
        scores_A_to_B = weight * N_cnt(A and B) / N_cnt(A) => P(B|A) = P(AB) / P(A)
    """

    group_by_col, agg_col = 'user_id', 'item_id'
 
    data_ = click_all.groupby(['user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()

    hot_list = list(click_all['item_id'].value_counts().index[:].values)
    stat_cnt = Counter(list(click_all['item_id'])) 
    stat_length = np.mean([ len(item_txt.split(',')) for item_txt in data_['item_id']])
  
    matrix_association_rules = {}
    print('------- association rules matrix 生成 ---------')
    for i, row in tqdm(data_.iterrows()):
        
        list_item_id = row['item_id'].split(',')
        len_list_item = len(list_item_id)

        for i, item_i in enumerate(list_item_id):
            for j, item_j in enumerate(list_item_id):

                if i <= j:
                    if item_i not in matrix_association_rules:
                            matrix_association_rules[item_i] = {}
                    if item_j not in matrix_association_rules[item_i]:
                            matrix_association_rules[item_i][item_j] = 0
                    
                    alpha, beta, gama = 1.0, 0.8, 0.8
                    matrix_association_rules[item_i][item_j] += 1.0 * alpha  / (beta + np.abs(i-j)) * 1.0 / stat_cnt[item_i] * 1.0 / (1 + gama * len_list_item / stat_length)
                if i >= j:
                    if item_i not in matrix_association_rules:
                        matrix_association_rules[item_i] = {}
                    if item_j not in matrix_association_rules[item_i]:
                        matrix_association_rules[item_i][item_j] = 0
                    
                    alpha, beta, gama = 0.5, 0.8, 0.8
                    matrix_association_rules[item_i][item_j] += 1.0 * alpha  / (beta + np.abs(i-j)) * 1.0 / stat_cnt[item_i] * 1.0 / (1 + gama * len_list_item / stat_length)
         
    # print(len(matrix_association_rules.keys()))
    # print(len(set(click_all['item_id'])))
    # print('data - matrix: ')
    # print( set(click_all['item_id']) - set(matrix_association_rules.keys()) )
    # print('matrix - data: ')
    # print( set(matrix_association_rules.keys()) - set(click_all['item_id']))
    assert len(matrix_association_rules.keys()) == len(set(click_all['item_id']))

    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- association rules 召回 ---------')
    for i, row in tqdm(data_.iterrows()):
        
        list_item_id = row['item_id'].split(',')

        dict_item_id_score = {}
        for i, item_i in enumerate(list_item_id[::-1]):
            for item_j, score_similar in sorted(matrix_association_rules[item_i].items(), reverse=True)[0:k]:
                if item_j not in list_item_id:
                    if item_j not in dict_item_id_score:
                        dict_item_id_score[item_j] = 0
                    sigma = 0.8
                    dict_item_id_score[item_j] +=  1.0 / (1 + sigma * i) * score_similar

        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
 
        # 不足的热度补全
        if len(dict_item_id_score_topk) < k:
            for i, item in enumerate(hot_list):
                if (item not in list_item_id) and (item not in dict_item_id_set):
                    item_similar = item
                    score_similar = - i - 100 
                    dict_item_id_score_topk.append( (item_similar, score_similar) )
                if len(dict_item_id_score_topk) == k:
                    break

        assert len(dict_item_id_score_topk) == k
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])
            
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall, matrix_association_rules




def topk_recall_glove_embedding(click_all,
                      dict_label,
                      k=100,
                      dim=88,
                      epochs=30,
                      learning_rate=0.5
                      ):
    """
    glove embeddimg recall
    """

    data_ = click_all.groupby(['pred','user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()
    list_data = list(data_['item_id'].map(lambda x:x.split(',')))

    corpus_model = Corpus()
    corpus_model.fit(list_data, window=999999)
    
    glove = Glove(no_components=dim, learning_rate=learning_rate)
    glove.fit(corpus_model.matrix, epochs=epochs, no_threads=psutil.cpu_count(), verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- glove 召回 ---------')
    for i, row in tqdm(data_.iterrows()):
        
        list_item_id = row['item_id'].split(',')

        dict_item_id_score = {}
        for i, item in enumerate(list_item_id[::-1]):
            most_topk = glove.most_similar(item, number=k)
            for item_similar, score_similar in most_topk:
                if item_similar not in list_item_id:
                    if item_similar not in dict_item_id_score:
                        dict_item_id_score[item_similar] = 0
                    sigma = 0.8
                    dict_item_id_score[item_similar] += 1.0 / (1 + sigma * i) * score_similar
        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        assert len(dict_item_id_score_topk) == k
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])
            
    
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall




def topk_recall_word2vec_embedding(click_all,
                        dict_label,
                        k=100,
                        dim=88,
                        epochs=30,
                        learning_rate=0.5
                        ):
    
    """
    word2vec recall
    """
    data_ = click_all.groupby(['pred','user_id'])['item_id'].agg(lambda x:','.join(list(x))).reset_index()
    list_data = list(data_['item_id'].map(lambda x:x.split(',')))

    model = gensim.models.Word2Vec(
                    list_data,
                    size=dim,
                    alpha=learning_rate,
                    window=999999,
                    min_count=1,
                    workers=psutil.cpu_count(),
                    compute_loss=True,
                    iter=epochs,
                    hs=0,
                    sg=1,
                    seed=42
                )

    list_user_id = []
    list_item_similar = []
    list_score_similar = []
    print('------- word2vec 召回 ---------')
    for i, row in tqdm(data_.iterrows()):
        
        list_item_id = row['item_id'].split(',')

        dict_item_id_score = {}
        for i, item in enumerate(list_item_id[::-1]):
            most_topk = model.wv.most_similar(item, topn=k)
            for item_similar, score_similar in most_topk:
                if item_similar not in list_item_id:
                    if item_similar not in dict_item_id_score:
                        dict_item_id_score[item_similar] = 0
                    sigma = 0.8
                    dict_item_id_score[item_similar] += 1.0 / (1 + sigma * i) * score_similar
        dict_item_id_score_topk = sorted(dict_item_id_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
        assert len(dict_item_id_score_topk) == k
        dict_item_id_set = set([item_similar for item_similar, score_similar in dict_item_id_score_topk])
        assert len(dict_item_id_set) == k
        for item_similar, score_similar in dict_item_id_score_topk:
            list_item_similar.append(item_similar)
            list_score_similar.append(score_similar)
            list_user_id.append(row['user_id'])
            
    
    topk_recall = pd.DataFrame({'user_id':list_user_id,'item_similar':list_item_similar,'score_similar':list_score_similar})
    topk_recall['next_item_id'] = topk_recall['user_id'].map(dict_label)
    topk_recall['pred'] = topk_recall['user_id'].map(lambda x: 'train' if x in dict_label else 'test')

    return topk_recall