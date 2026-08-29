import pandas as pd
import csv
import ast
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = DATA_DIR / 'raw_data'
ORI_DATA_DIR = DATA_DIR / 'ori_data'
ORI_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 读取推荐训练数据
with open(RAW_DATA_DIR / 'recommendation_train.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    train_rec_df = pd.DataFrame(list(reader))
    train_rec_df.columns = train_rec_df.iloc[0]
    train_rec_df = train_rec_df.iloc[1:]

# 处理推荐训练数据
train_rec_result = []
for idx, row in train_rec_df.iterrows():
    user_idx = row['user_idx']
    rec_str = row['rec_result_details_with_idx']
    try:
        rec_list = re.findall(r"\{.*?\}", rec_str, re.DOTALL)
    except Exception as e:
        continue

    gt_note_list = []
    neg_note_list = []
    for rec in rec_list:
        rec_dict = ast.literal_eval(rec)
        if rec_dict.get('click', 0) == 1:
            note_idx = rec_dict.get('note_idx', None)
            timestamp = rec_dict.get('request_timestamp', None)
            gt_note_list.append(note_idx)
        else:
            note_idx = rec_dict.get('note_idx', None)
            timestamp = rec_dict.get('request_timestamp', None)
            neg_note_list.append(note_idx)
    history = row['recent_clicked_note_idxs']
    # change to list
    history = history.strip('[]').split()
    train_rec_result.append([user_idx, gt_note_list, neg_note_list, timestamp, history])

# 转为DataFrame
train_rec_result_df = pd.DataFrame(train_rec_result, columns=['user_idx', 'gt_note_idx', 'neg_note_idx', 'timestamp', 'history'])

# 处理推荐测试数据
with open(RAW_DATA_DIR / 'recommendation_test.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    test_rec_df = pd.DataFrame(list(reader))
    test_rec_df.columns = test_rec_df.iloc[0]
    test_rec_df = test_rec_df.iloc[1:]

test_rec_result = []
for idx, row in test_rec_df.iterrows():
    user_idx = row['user_idx']
    rec_str = row['rec_result_details_with_idx']
    try:
        rec_list = re.findall(r"\{.*?\}", rec_str, re.DOTALL)
    except Exception as e:
        continue

    gt_note_list = []
    neg_note_list = []
    for rec in rec_list:
        rec_dict = ast.literal_eval(rec)
        if rec_dict.get('click', 0) == 1:
            note_idx = rec_dict.get('note_idx', None)
            timestamp = rec_dict.get('request_timestamp', None)
            gt_note_list.append(note_idx)
        else:
            note_idx = rec_dict.get('note_idx', None)
            timestamp = rec_dict.get('request_timestamp', None)
            neg_note_list.append(note_idx)
    history = row['recent_clicked_note_idxs']
    # change to list
    history = history.strip('[]').split()
    test_rec_result.append([user_idx, gt_note_list, neg_note_list, timestamp, history])

# 转为DataFrame
test_rec_result_df = pd.DataFrame(test_rec_result, columns=['user_idx', 'gt_note_idx', 'neg_note_idx', 'timestamp', 'history'])

# merge the train and test data
rec_result_df = pd.concat([train_rec_result_df, test_rec_result_df])

# 处理搜索训练数据
with open(RAW_DATA_DIR / 'search_train.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    search_df = pd.DataFrame(list(reader))
    search_df.columns = search_df.iloc[0]
    search_df = search_df.iloc[1:]

train_search_result = []
for idx, row in search_df.iterrows():
    user_idx = row['user_idx']
    search_str = row['search_result_details_with_idx']
    search_list = re.findall(r"\{.*?\}", search_str, re.DOTALL)

    gt_note_list = []
    neg_note_list = []
    for search in search_list:
        search = search.replace('nan', 'None') 
        try:
            search_dict = ast.literal_eval(search)
        except Exception as e:
            print(f"解析失败内容：{search}\n错误：{e}")
            continue
        if int(search_dict['click']) == 1:
            note_idx = search_dict.get('note_idx', None)
            timestamp = search_dict.get('search_timestamp', None)
            query = row['query']
            gt_note_list.append(note_idx)
        else:
            note_idx = search_dict.get('note_idx', None)
            timestamp = search_dict.get('search_timestamp', None)
            query = row['query']
            neg_note_list.append(note_idx)
    history = row['recent_clicked_note_idxs']
    # change to list
    history = history.strip('[]').split()
    train_search_result.append([user_idx, gt_note_list, neg_note_list, timestamp, query, history])

train_search_result_df = pd.DataFrame(train_search_result, columns=['user_idx', 'gt_note_idx', 'neg_note_idx', 'timestamp', 'query', 'history'])

# 处理搜索测试数据
with open(RAW_DATA_DIR / 'search_test.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    test_search_df = pd.DataFrame(list(reader))
    test_search_df.columns = test_search_df.iloc[0]
    test_search_df = test_search_df.iloc[1:]

test_search_result = []
for idx, row in test_search_df.iterrows():
    user_idx = row['user_idx']
    search_str = row['search_result_details_with_idx']
    try:
        search_list = re.findall(r"\{.*?\}", search_str, re.DOTALL)
    except Exception as e:
        continue

    gt_note_list = []
    neg_note_list = []
    for search in search_list:
        search = search.replace('nan', 'None') 
        try:
            search_dict = ast.literal_eval(search)
        except Exception as e:
            print(f"解析失败内容：{search}\n错误：{e}")
            continue

        if search_dict.get('click', 0) == 1:
            note_idx = search_dict.get('note_idx', None)
            timestamp = search_dict.get('search_timestamp', None)
            query = row['query']
            gt_note_list.append(note_idx)
        else:
            note_idx = search_dict.get('note_idx', None)
            timestamp = search_dict.get('search_timestamp', None)
            query = row['query']
            neg_note_list.append(note_idx)
    history = row['recent_clicked_note_idxs']
    # change to list
    history = history.strip('[]').split()
    test_search_result.append([user_idx, gt_note_list, neg_note_list, timestamp, query, history])

test_search_result_df = pd.DataFrame(test_search_result, columns=['user_idx', 'gt_note_idx', 'neg_note_idx', 'timestamp', 'query', 'history'])

# merge the train and test data
search_result_df = pd.concat([train_search_result_df, test_search_result_df])

# add a column 'task' to both rec_result_df and search_result_df
rec_result_df['task'] = 'rec'
search_result_df['task'] = 'search'

# only keep the users and items that have at least 3 interactions for both rec and search
rec_result_df = rec_result_df[rec_result_df['user_idx'].isin(rec_result_df['user_idx'].value_counts()[rec_result_df['user_idx'].value_counts() >= 3].index)]
search_result_df = search_result_df[search_result_df['user_idx'].isin(search_result_df['user_idx'].value_counts()[search_result_df['user_idx'].value_counts() >= 3].index)]

# merge the two dataframes
result_df = pd.concat([rec_result_df, search_result_df])

# sort by timestamp from old to new
result_df = result_df.sort_values(by='timestamp', ascending=True)

# only keep user who has both search and rec behavior
rec_users = set(result_df[result_df['task'] == 'rec']['user_idx'].unique())
search_users = set(result_df[result_df['task'] == 'search']['user_idx'].unique())
both_users = rec_users & search_users
result_df = result_df[result_df['user_idx'].isin(both_users)]

# 循环过滤直到user和item都达到3-core
print("开始循环过滤直到user和item都达到3-core...")
max_iterations = 10
for iteration in range(max_iterations):
    print(f"第 {iteration + 1} 次迭代:")
    
    # 统计当前user的交互次数
    user_counts = result_df['user_idx'].value_counts()
    valid_users = set(user_counts[user_counts >= 3].index)
    print(f"  有效用户数量: {len(valid_users)}")
    
    # 过滤user
    result_df = result_df[result_df['user_idx'].isin(valid_users)]
    
    # 统计当前item的交互次数
    all_items = []
    for _, row in result_df.iterrows():
        all_items.extend(row['gt_note_idx'])
        all_items.extend(row['neg_note_idx'])
        all_items.extend(row['history'])
    
    item_counts = pd.Series(all_items).value_counts()
    valid_items = set(item_counts[item_counts >= 3].index)
    print(f"  有效item数量: {len(valid_items)}")
    
    # 过滤item
    def filter_items(item_list):
        if isinstance(item_list, list):
            return [item for item in item_list if item in valid_items]
        return item_list
    
    result_df['gt_note_idx'] = result_df['gt_note_idx'].apply(filter_items)
    result_df['neg_note_idx'] = result_df['neg_note_idx'].apply(filter_items)
    result_df['history'] = result_df['history'].apply(filter_items)
    
    
    # 移除没有有效item的行
    result_df = result_df[
        (result_df['gt_note_idx'].apply(len) > 0) | 
        (result_df['neg_note_idx'].apply(len) > 0)
    ]
    
    print(f"  当前数据量: {len(result_df)}")
    
    # 检查是否收敛
    current_user_count = len(valid_users)
    current_item_count = len(valid_items)
    
    if iteration > 0:
        if current_user_count == prev_user_count and current_item_count == prev_item_count:
            print(f"  收敛！在第 {iteration + 1} 次迭代后停止")
            break
    
    prev_user_count = current_user_count
    prev_item_count = current_item_count

print(f"最终数据量: {len(result_df)}")
print(f"最终用户数量: {len(result_df['user_idx'].unique())}")

# 重新统计最终item数量
all_items = []
for _, row in result_df.iterrows():
    all_items.extend(row['gt_note_idx'])
    all_items.extend(row['neg_note_idx'])
final_item_counts = pd.Series(all_items).value_counts()
print(f"最终item数量: {len(final_item_counts)}")

# split into train/valid/test according to leave-one-out strategy
rec_result_df = result_df[result_df['task'] == 'rec'].copy()
rec_result_df['user_idx'] = rec_result_df['user_idx'].astype(int)

search_result_df = result_df[result_df['task'] == 'search'].copy()
search_result_df['user_idx'] = search_result_df['user_idx'].astype(int)

def split_by_user(df):
    train, valid, test = [], [], []
    for user, user_df in df.groupby('user_idx'):
        user_df = user_df.sort_values('timestamp')
        if len(user_df) >= 3:
            test.append(user_df.iloc[-1])
            valid.append(user_df.iloc[-2])
            train.append(user_df.iloc[:-2])
    train_df = pd.concat(train).reset_index(drop=True) if train else pd.DataFrame(columns=df.columns)
    valid_df = pd.DataFrame(valid).reset_index(drop=True) if valid else pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(test).reset_index(drop=True) if test else pd.DataFrame(columns=df.columns)
    return train_df, valid_df, test_df

# 用法
train_rec_df, valid_rec_df, test_rec_df = split_by_user(rec_result_df)
train_search_df, valid_search_df, test_search_df = split_by_user(search_result_df)

train_df = pd.concat([train_rec_df, train_search_df])
valid_df = pd.concat([valid_rec_df, valid_search_df])
test_df = pd.concat([test_rec_df, test_search_df])

# print statistics of the train/valid/test dataframes
print(f"Train data: {len(train_df)}")
print(f"Valid data: {len(valid_df)}")
print(f"Test data: {len(test_df)}")

# for search/rec, also print the data num
print(f"Train rec data: {len(train_rec_df)}")
print(f"Train search data: {len(train_search_df)}")
print(f"Valid rec data: {len(valid_rec_df)}")
print(f"Valid search data: {len(valid_search_df)}")
print(f"Test rec data: {len(test_rec_df)}")
print(f"Test search data: {len(test_search_df)}")

# Calculate unique users and items across all datasets
all_users = pd.concat([train_df, valid_df, test_df])['user_idx'].unique()
all_items = pd.concat([train_df, valid_df, test_df])['gt_note_idx'].explode().unique()

print(f"Total unique users: {len(all_users)}")
print(f"Total unique items: {len(all_items)}")

# save to pkl
train_df.to_pickle(ORI_DATA_DIR / 'train.pkl')
valid_df.to_pickle(ORI_DATA_DIR / 'valid.pkl')
test_df.to_pickle(ORI_DATA_DIR / 'test.pkl')
train_rec_df.to_pickle(ORI_DATA_DIR / 'rec_train.pkl')
train_search_df.to_pickle(ORI_DATA_DIR / 'src_train.pkl')
valid_rec_df.to_pickle(ORI_DATA_DIR / 'rec_valid.pkl')
valid_search_df.to_pickle(ORI_DATA_DIR / 'src_valid.pkl')
test_rec_df.to_pickle(ORI_DATA_DIR / 'rec_test.pkl')
test_search_df.to_pickle(ORI_DATA_DIR / 'src_test.pkl')

print(f"数据处理完成，已保存到 {ORI_DATA_DIR}") 
