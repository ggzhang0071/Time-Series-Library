import os
import pandas as pd 
from sklearn.impute import KNNImputer


def merge_target(root_path,data_path,target):
    df=pd.read_csv(os.path.join(root_path,data_path))
    if '4500K1.0S' or "5000K0.8S" or '5500K0.8S' in df.columns:
        df['target']=(df["4500K1.0S"]+df["5000K0.8S"])/2
        df=df.drop('4500K1.0S',axis=1)
        df=df.drop('5000K0.8S',axis=1)
        df=df.drop('5500K0.8S',axis=1)  
        df.rename(columns={"日期":"date"},inplace=True)
        file_name, ext=os.path.splitext(data_path)  
        data_path_new_new=file_name+"_"+target+ext
        print(f"save file name: {data_path_new}")
        df.to_csv(os.path.join(root_path,data_path_new), index=False)

def choose_variable(root_path,data_path,all_targets,target):
    df=pd.read_csv(os.path.join(root_path,data_path))
    all_targets.remove(target)
    for column in all_targets:
        if column in df.columns:
            df=df.drop(column,axis=1)
    df.rename(columns={"日期":"date"},inplace=True)
    file_name, ext=os.path.splitext(data_path)  
    data_path_new=file_name+"_"+target+ext
    print(f"save file name: {data_path_new}")
    df.to_csv(os.path.join(root_path,data_path_new), index=False)

def prepare_data_for_TSL(args,dir_path,file_path):
    df_raw = pd.read_csv(os.path.join(dir_path,file_path))

    # 对缺失值进行处理
    if df_raw.isnull().values.any():
        imputer = KNNImputer(n_neighbors=10) 
        for column in df_raw.columns:
            if column != 'date':  # 排除不需要插值的时间戳列
                df_raw[[column]] = imputer.fit_transform(df_raw[[column]])

    # 是否对目标数据y 进行处理
    if args.target_preprocess=="diff":
        df_raw[args.target]= df_raw[args.target].diff()
        df_raw = df_raw.drop(df_raw.index[0])
    elif args.target_preprocess=="original":
        target_value=df_raw[args.target]
        mean_value = target_value.mean()
        target_value = (target_value >= mean_value).astype(int)
        df_raw[args.target]=target_value
    else:
        raise Exception("The target preprocess isn't existed")

    # 移除 data 和target 项
    cols = list(df_raw.columns)
    cols.remove(args.target)
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + [args.target]]
    df_raw_len=int(len(df_raw))

    # split train, val and test 
    num_train = int(df_raw_len* 0.8)
    num_test = int(df_raw_len* 0.1)  
    num_vali = df_raw_len - num_train - num_test

    # 拆分数据
    train_data = df_raw[:num_train]
    vali_data = df_raw[num_train:num_train + num_vali]
    test_data = df_raw[num_train + num_vali:]

    # 保存拆分后的数据
    train_filename = file_path.replace('.csv', '_TRAIN.csv')
    vali_filename = file_path.replace('.csv', '_VAL.csv')
    test_filename = file_path.replace('.csv', '_TEST.csv')
    new_dir_path=os.path.join(dir_path,args.target)
    os.makedirs(new_dir_path, exist_ok=True) if not os.path.exists(new_dir_path) else None
    
    train_data.to_csv(os.path.join(new_dir_path,train_filename), index=False)
    vali_data.to_csv(os.path.join(new_dir_path,vali_filename), index=False)
    test_data.to_csv(os.path.join(new_dir_path,test_filename), index=False)

    print(f"Training data saved to {train_filename}")
    print(f"Validation data saved to {vali_filename}")
    print(f"Test data saved to {test_filename}")
    

if __name__=="__main__":
    root_path='/git/datasets/beigang_data'
    #data_path='all_variables_for_mine_price.csv'
    #data_path_new='all_variables_for_mine_price_merged.csv'
    data_path="runmin_an_factors.csv"
    all_targets=["4500K1.0S",'5000K0.8S','5500K0.8S']
    target=all_targets[2]
    if target in all_targets:
        choose_variable(root_path,data_path,all_targets,target)
    elif target=="merged":
        merge_target(root_path,data_path,target)

    class Args:
        def __init__(self,target):
            self.target_preprocess = 'for_classification'
            self.target=target
    args=Args(all_targets[0])
    csv_path="/git/datasets/beigang_data"
    file_path="runmin_an_factors_4500K1.0S.csv"
    prepare_data_for_TSL(args,csv_path,file_path)


    

    

      





