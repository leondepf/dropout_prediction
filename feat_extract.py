import pandas as pd

train = pd.read_csv('D:/Datasets/MOOC数据集/KDD2015/prediction_log/train_log.csv')
test = pd.read_csv('D:/Datasets/MOOC数据集/KDD2015/prediction_log/test_log.csv')
train_truth = pd.read_csv('D:/Datasets/MOOC数据集/KDD2015/prediction_log/train_truth.csv', index_col='enroll_id')
test_truth = pd.read_csv('D:/Datasets/MOOC数据集/KDD2015/prediction_log/test_truth.csv', index_col='enroll_id')
all_truth = pd.concat([train_truth, test_truth])
all_log = pd.concat([train, test])

train_enroll = list(set(list(train['enroll_id'])))
#把train_log.csv数据集中的enroll_id字段下数据组成一个list，去除重复值(set方法)后再组成一个list

test_enroll = list(set(list(test['enroll_id'])))
video_action = ['seek_video','play_video','pause_video','stop_video','load_video']
problem_action = ['problem_get','problem_check','problem_save','reset_problem','problem_check_correct', 'problem_check_incorrect']
forum_action = ['create_thread','create_comment','delete_thread','delete_comment']
click_action = ['click_info','click_courseware','click_about','click_forum','click_progress']
close_action = ['close_courseware']

all_num = all_log.groupby('enroll_id').count()[['action']]  #根据enroll_id对action字段计数
all_num.columns = ['all#count']  #计数结果存入all#count的新字段中
session_enroll = all_log[['session_id']].drop_duplicates()  #去除重复项
session_num = all_log.groupby('enroll_id').count()
all_num['session#count'] = session_num['session_id']
for a in video_action + problem_action + forum_action + click_action + close_action:
    action_ = (all_log['action'] == a).astype(int)  #结果为0或1
    all_log[a+'#num'] = action_      #对应行为如video_action#num为0或1
    #action_num = all_log.groupby('enroll_id').sum()[[a+'#num']]
    action_num = all_log.groupby(['enroll_id'])[a+'#num'].sum()
    all_num = pd.merge(all_num, action_num, left_index=True, right_index=True)
all_num = pd.merge(all_num, all_truth, left_index=True, right_index=True)
enroll_info = all_log[['username','course_id','enroll_id']].drop_duplicates()
enroll_info.index = enroll_info['enroll_id']
del enroll_info['enroll_id']
all_num = pd.merge(all_num, enroll_info, left_index=True, right_index=True)
all_num.loc[test_enroll].to_csv('test_features.csv')
all_num.loc[train_enroll].to_csv('train_features.csv')

