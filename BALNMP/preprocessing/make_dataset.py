### annotation
'''
json 형태를 보면, 
10개씩 patch 묶어서 하나의 label로 전달 (bag)
단, "id"의 모든 patch들을 겹치지 않게 10개씩 최대한으로 구성하는 듯
예를들어, "id": "981"환자를 보면 총 패치개수는 76개, 10개씩 7개의 bag만 사용 (즉 70개의 패치를 10개의 bag으로 만듬)

=> 만약에 bag에 들어가지 못한 6개 패치중 중요한 패치가 있었다면? 아예 안들어가는게 아닌가.. 그래서 최대한 복원추출해야 하는게 아닌가..생각?
'''

# 1,0의 분포는 거의 반반 => 486 / 514
# train 환자 1000 이므로 600 / 200 / 200 (train / val / test) split == 60% 20% 20%
# test set (200)
## 0: 100개
## 1: 100개

# train set (600)
## 0: 300개
## 1: 300개

# val set (200)
## 0: 100개
## 1: 100개

import os
import pandas as pd
import random
from tqdm import tqdm
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 100 # split_patients()
NAN_NUM = -1 # make_clinical_data()

def split_patients():
    csv_path = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train.csv'
    
    folds = {}
    anno_df = pd.read_csv(csv_path, encoding='cp949')

    # split test    
    data = anno_df['ID']
    target = anno_df['N_category']
    x_train, x_test, _, _ = train_test_split(data, target, test_size=0.2, shuffle=True, stratify=target, random_state=RANDOM_SEED)
    test_df = anno_df.loc[x_test.index]
    
    # split train, val (kfold=4)
    train_df = anno_df.loc[x_train.index]
    
    skf = StratifiedKFold(n_splits=4,random_state=RANDOM_SEED,shuffle=True)
    
    for i, (train_index, valid_index) in enumerate(skf.split(train_df['ID'], train_df['N_category']), 0):
        folds[i] = {
            'train':train_df.iloc[train_index],
            'valid':train_df.iloc[valid_index],
            'test': test_df
            }
    
    
    # print('@@@ train \n', folds[1]['train'], '\n')
    # print('@@@ valid \n', folds[1]['valid'], '\n')
    # print('@@@ test \n', test_df['ID'], '\n')
    
    return folds

def make_clinical_data():
    csv_path = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/test.csv' # target csv path
    anno_df = pd.read_csv(csv_path, encoding='cp949')
    
    train_csv_path = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train.csv'
    train_anno_df = pd.read_csv(train_csv_path, encoding='cp949')
    
    clinical_dir = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/clinical_data_new'
    os.makedirs(clinical_dir, exist_ok=True)
    
    # nan 처리
    anno_df = anno_df.fillna(NAN_NUM)
    train_anno_df = train_anno_df.fillna(NAN_NUM)
    
    # set columns
    '''
        ['ID', 'img_path', 'mask_path', '나이', '수술연월일', '진단명', '암의 위치', '암의 개수',
        '암의 장경', 'NG', 'HG', 'HG_score_1', 'HG_score_2', 'HG_score_3',
        'DCIS_or_LCIS_여부', 'DCIS_or_LCIS_type', 'T_category', 'ER',
        'ER_Allred_score', 'PR', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2',
        'HER2_IHC', 'HER2_SISH', 'HER2_SISH_ratio', 'BRCA_mutation',
        'N_category']
    '''
    clinical_col = ['ID', '나이', '진단명', '암의 위치', '암의 개수', '암의 장경', 'NG', 'HG', \
        'HG_score_1', 'HG_score_2', 'HG_score_3', 'DCIS_or_LCIS_여부', 'DCIS_or_LCIS_type', \
            'T_category', 'ER', 'ER_Allred_score', 'PR', 'PR_Allred_score', 'KI-67_LI_percent', \
                'HER2', 'HER2_IHC', 'HER2_SISH', 'HER2_SISH_ratio', 'BRCA_mutation']
    anno_df = anno_df[clinical_col]
    train_anno_df = train_anno_df[clinical_col]
    
    # standardization
    # first, calc standaridization param on trainset
    train_value = train_anno_df.iloc[:, 1:].values
    train_scaler = StandardScaler()
    train_scaler.fit(train_value) # train standaridization param

    # second, apply trainset param
    anno_value = anno_df.iloc[:, 1:].values
    anno_value = train_scaler.transform(anno_value)
    anno_df.iloc[:, 1:] = anno_value

    # save
    anno_df.to_excel(os.path.join(clinical_dir, 'clinical_test_data.xlsx'), index=False)


def make_bags():
    folds = split_patients()

    BAG_SIZE = 10
    
    patches_root_dir  = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/train_imgs_PyHIST'
    bags_dir = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json'
    os.makedirs(bags_dir, exist_ok=True)
    
    for f_num, fold in tqdm(folds.items()):
        # train_df = fold['train']
        # valid_df = fold['valid']
        # test_df = fold['test']
        
        bags = []
        
        for df_key in ['train', 'valid', 'test']:
            print('fold {} - {} ... '.format(f_num, df_key))
            
            target_df = fold[df_key]
            
            for i in range(len(target_df)):
                p_id = target_df.iloc[i, 0]
                p_label = target_df.iloc[i, -1]
                
                # search patches
                patches_dir = os.path.join(patches_root_dir, p_id, '{}_tiles'.format(p_id))
                p_patches = os.listdir(patches_dir)
                random.Random(RANDOM_SEED).shuffle(p_patches)
                
                b_cnt, _ = divmod(len(p_patches), BAG_SIZE)
                
                if b_cnt == 0:
                    print('continue')
                    continue
                
                # insert bag on bags
                for b_num in range(b_cnt):
                    pp = p_patches[BAG_SIZE * b_num : BAG_SIZE * (b_num+1)]
                    
                    bag = {
                        'id': p_id,
                        'label': str(p_label),
                        'patch_paths': [os.path.join(p_id, '{}_tiles'.format(p_id), p) for p in pp]
                    }
                    
                    bags.append(bag)
            
            with open(os.path.join(bags_dir, '{}-fold-{}.json'.format(df_key, f_num)), "w") as outfile:
                json.dump(bags, outfile, indent=4)
            
def make_infer_bags():
    csv_path = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/test.csv'
    target_df = pd.read_csv(csv_path, encoding='cp949')
    
    BAG_SIZE = 10
    
    patches_root_dir  = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/test_imgs_PyHIST'
    bags_dir = '/mnt/d/data_ai/challenge/breast_cancer_metastasis/assets/open/dataset/json'
    os.makedirs(bags_dir, exist_ok=True)
       
    bags = []
    
    for i in range(len(target_df)):
        p_id = target_df.iloc[i, 0]
        
        # search patches
        patches_dir = os.path.join(patches_root_dir, p_id, '{}_tiles'.format(p_id))
        p_patches = os.listdir(patches_dir)
        random.Random(RANDOM_SEED).shuffle(p_patches)
        
        b_cnt, _ = divmod(len(p_patches), BAG_SIZE)
        
        if b_cnt == 0:
            print('continue')
            continue
        
        # insert bag on bags
        for b_num in range(b_cnt):
            pp = p_patches[BAG_SIZE * b_num : BAG_SIZE * (b_num+1)]
            
            bag = {
                'id': p_id,
                'patch_paths': [os.path.join(p_id, '{}_tiles'.format(p_id), p) for p in pp]
            }
            
            bags.append(bag)
    
    with open(os.path.join(bags_dir, 'inference.json'), "w") as outfile:
        json.dump(bags, outfile, indent=4)

if __name__ == "__main__":
    # split_patients()
    # make_clinical_data()
    # make_bags() # train
    make_infer_bags() # test (inference)