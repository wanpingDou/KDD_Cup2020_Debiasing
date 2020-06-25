#coding:utf-8

def metrics_recall(topk_recall, phase, k, sep=10):
    """
    recall metric
    """
    data_ = topk_recall[topk_recall['pred']=='train'].sort_values(['user_id','score_similar'],ascending=False)
    data_ = data_.groupby(['user_id']).agg({'item_similar':lambda x:list(x),'next_item_id':lambda x:''.join(set(x))})

    data_['index'] = [recall_.index(label_) if label_ in recall_ else -1 for (label_, recall_) in zip(data_['next_item_id'],
                                                                                                      data_['item_similar'])]

    print('-------- 召回效果 -------------')
    print('--------:phase: ', phase,' -------------')
    data_num = len(data_)
    for topk in range(0,k+1,sep):
        hit_num = len(data_[(data_['index']!=-1) & (data_['index']<=topk)]) 
        hit_rate = hit_num * 1.0 / data_num
        print('phase: ', phase, ' top_', topk, ' : ', 'hit_num : ', hit_num, 'hit_rate : ', hit_rate, ' data_num : ', data_num)
        print() 

    hit_rate = len(data_[data_['index']!=-1]) * 1.0 / data_num
    return hit_rate