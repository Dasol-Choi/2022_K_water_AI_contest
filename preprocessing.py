#DataProcessing
import numpy as np
import pandas as pd

def processing(path):
    print('Welcome')

    train_df = pd.read_csv(path + 'data_tr_apo.csv')
    train_df.columns = ['datetime', 'D']

    test_df = pd.read_csv(path + 'data_ts_apo.csv')
    test_df.columns = ['datetime', 'D']

    submission = pd.read_csv(path + 'sample_apo.csv')

    # datetime 열을 datetime 객체로 변환
    train_df['datetime'] = pd.to_datetime(train_df['datetime'])
    test_df['datetime'] = pd.to_datetime(test_df['datetime'])
    submission ['datetime'] = pd.to_datetime(submission ['datetime'])

    # new features 생성 (년, 월, 일, 시간 분리)
    train_df['year'] = train_df['datetime'].dt.year
    train_df['month'] = train_df['datetime'].dt.month
    train_df['day'] = train_df['datetime'].dt.day
    train_df['hour'] = train_df['datetime'].dt.hour

    test_df['year'] = test_df['datetime'].dt.year
    test_df['month'] = test_df['datetime'].dt.month
    test_df['day'] = test_df['datetime'].dt.day
    test_df['hour'] = test_df['datetime'].dt.hour
    
    train_df.loc[train_df.index == 11993 ,'D'] = 82 + (118-82)/4*1
    train_df.loc[train_df.index == 11994 ,'D'] = 82 + (118-82)/4*2
    train_df.loc[train_df.index == 11995 ,'D'] = 82 + (118-82)/4*3

    train_df.loc[train_df.index == 2127 ,'D'] = int(87 + (115-87)/3*1)
    train_df.loc[train_df.index == 2128 ,'D'] = int(87 + (115-87)/3*2)

    train_df.loc[train_df.index == 16193 ,'D'] = 88
    train_df.loc[train_df.index == 16194 ,'D'] = 88
    train_df.loc[train_df.index == 16195 ,'D'] = 88
    train_df.loc[train_df.index == 17558 ,'D'] = (110 + 88) /2
    train_df.loc[train_df.index == 19310 ,'D'] = 93
    train_df.loc[train_df.index == 19311 ,'D'] = 93
    train_df.loc[train_df.index == 19312 ,'D'] = 93

    train_df.loc[train_df.index == 4484 ,'D'] = int(85 + (129 - 85)/3*1)
    train_df.loc[train_df.index == 4485 ,'D'] =int(85 + (129 - 85)/3*2)

    train_df.loc[train_df.index == 13234 ,'D'] = 94
    train_df.loc[train_df.index == 13235 ,'D'] = 94

    train_df.loc[train_df.index == 18590 ,'D'] = (108 + 92) /2
    train_df.loc[train_df.index == 32226 ,'D'] = (245 + 99) /2
    train_df.loc[train_df.index == 33396 ,'D'] = (104 + 99) /2
    train_df.loc[train_df.index == 9544 ,'D'] = (142 + 93) /2
    train_df.loc[train_df.index == 26463 ,'D'] = (104 + 89) /2
    train_df.loc[train_df.index == 29097 ,'D'] = (148 + 117) /2

    train_df.loc[train_df.index == 10205 ,'D'] = int(35 + (106 - 35)/5*1)
    train_df.loc[train_df.index == 10206 ,'D'] = int(35 + (106 - 35)/5*2)
    train_df.loc[train_df.index == 10207 ,'D'] = int(35 + (106 - 35)/5*3)
    train_df.loc[train_df.index == 10208 ,'D'] = int(35 + (106 - 35)/5*4)

    train_df.loc[train_df.index == 4523 ,'D'] = int(80 + (148 - 80)/5*1)
    train_df.loc[train_df.index == 4524 ,'D'] = int(80 + (148 - 80)/5*2)
    train_df.loc[train_df.index == 4525 ,'D'] = int(80 + (148 - 80)/5*3)
    train_df.loc[train_df.index == 4526 ,'D'] = int(80 + (148 - 80)/5*4)

    train_df.loc[train_df.index == 16554 ,'D'] = (95 + 61)/2
    train_df.loc[train_df.index == 34402 ,'D'] = 250
    train_df.loc[train_df.index == 29842 ,'D'] = 250

    train_df.loc[train_df.index == 25666 ,'D'] = 240
    train_df.loc[train_df.index == 25667 ,'D'] = 250

    train_df.loc[train_df.index == 12179 ,'D'] = 250

    train_df.loc[train_df.index == 16091 ,'D'] = 250
    train_df.loc[train_df.index == 16091 ,'D'] = 240

    train_df.loc[train_df.index == 20962 ,'D'] = 240
    train_df.loc[train_df.index == 20963 ,'D'] = 250
    train_df.loc[train_df.index == 20964 ,'D'] = 240

    train_df.loc[train_df.index == 15178 ,'D'] = 250
    train_df.loc[train_df.index == 16092 ,'D'] = 250
    train_df.loc[train_df.index == 20963 ,'D'] = 240
    train_df.loc[train_df.index == 25667,'D'] = 250
    train_df.loc[train_df.D >= 250, 'D'] = 250
    train_df.loc[train_df.D <= 40, 'D'] = 45

    train_df.loc[train_df.index == 1619,'D'] = 200
    train_df.loc[train_df.index == 1620,'D'] = 200
    train_df.loc[train_df.index == 1621,'D'] = 200

    train_df.loc[train_df.index == 1640,'D'] = 190
    train_df.loc[train_df.index == 1641,'D'] = 190
    train_df.loc[train_df.index == 1642,'D'] = 190
    train_df.loc[train_df.index == 1643,'D'] = 200
    train_df.loc[train_df.index == 1644,'D'] = 200
    train_df.loc[train_df.index == 1645,'D'] = 200

    train_df.loc[(train_df.year == 2017) & (train_df.D > 230), 'D'] = 200

    train_df.loc[train_df.index == 1748,'D'] = (91+145)/2
    train_df.loc[train_df.index == 1982,'D'] = (85+174)/2

    train_df.loc[train_df.index == 12716,'D'] = int(71 + (103-71)/3*1)
    train_df.loc[train_df.index == 12717,'D'] = int(71 + (103-71)/3*2)

    train_df.loc[train_df.index == 25934,'D'] = int(90 + (98-90)/3*1)
    train_df.loc[train_df.index == 25935,'D'] = int(90 + (98-90)/3*2)

    train_df.loc[train_df.index == 27253,'D'] = int(83 + (97-83)/3*1)
    train_df.loc[train_df.index == 27254,'D'] = int(83 + (97-83)/3*2)

    train_df.loc[train_df.index == 13187,'D'] = (105+110)/2
    
    
    
    # datetime을 index로 지정
    train_df = train_df.set_index('datetime')
    test_df = test_df.set_index('datetime')

    return train_df, test_df

#함수 선언 순서 주의
def make_sequene_dataset(feature, label, window_size):
    feature_list = []     
    label_list = [] 
    for i in range(len(feature)-window_size):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])
    return np.array(feature_list), np.array(label_list)

def data_form_descision(train_df, test_df, pre_day):# 전처리한 data, 예측에 사용할 month 기준 input으로 받기
    #제출 형식 맞추기 위한 병합
    X = pd.concat([train_df, test_df])
    for i in range(0,336):
        X[f'h{i}'] = 0
    X['h0'] = X['D']
    for i in range(1, 336):
        X[f'h{i}'] = X['D'].shift(-i)
    X.loc['2021-12-18 00:00:00'] = np.nan
    X.loc['2021-12-18 00:00:00'].year = 2021
    X.loc['2021-12-18 00:00:00'].month = 12
    X.loc['2021-12-18 00:00:00'].day = 18
    X.loc['2021-12-18 00:00:00'].hour = 0
    
    #test label로 사용 가능한 범위
    temp = X[(X.year==2021) & (X.month==12) & (X.day >= 4 )]
    
    #학습 frame과 test frame 나누기
    TRAIN = X.drop(temp.index)
    TEST = TRAIN[TRAIN.year == 2021]
    TRAIN = TRAIN.drop(TEST.index)
    
    #예측에 사용할 데이터 범위 결정
    temp1 = X[(X.year >= 2020) & (X.month == 12) & (X.day >= pre_day)] 
    temp2 = X[(X.year == 2021) & (X.month >= 1)] 
    
    #Train data와 test data 부분 합치기
    submission_form = pd.concat([temp1, temp2])
    
    #feature와 label column구분
    feature_cols = ['D']
    label_cols = [f'h{i}' for i in range(336)]
    train_feature_df = pd.DataFrame(TRAIN, columns=feature_cols)
    train_label_df = pd.DataFrame(TRAIN, columns=label_cols)
    test_feature_df = pd.DataFrame(TEST, columns=feature_cols)
    test_label_df = pd.DataFrame(TEST, columns=label_cols)
    submission_feature_df = pd.DataFrame(submission_form, columns=feature_cols)
    submission_label_df = pd.DataFrame(submission_form, columns=label_cols)
    
    # 2017년 데이터 앞에 일부 빼기
    train_feature_df = train_feature_df[7600:]
    train_label_df = train_label_df[7600:]
    
    #numpy로 변환
    train_feature_np = train_feature_df.to_numpy()
    train_label_np = train_label_df.to_numpy()
    test_feature_np = test_feature_df.to_numpy()
    test_label_np = test_label_df.to_numpy()
    submission_feature_np = submission_feature_df.to_numpy()
    submission_label_np = submission_label_df.to_numpy()
    
    #window_size에 맞게 데이터 shape 조정
    window_size = submission_form.shape[0] - 8425 #submission_form.shape[0] is max_size // 8425 is submission의 개수
    x_train, y_train = make_sequene_dataset(train_feature_np, train_label_np, window_size)
    x_test, y_test = make_sequene_dataset(test_feature_np, test_label_np, window_size)
    submission_x, submission_y = make_sequene_dataset(submission_feature_np, submission_label_np, window_size)
    
    return x_train, y_train, x_test, y_test, submission_x, submission_y


