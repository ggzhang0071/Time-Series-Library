import os
import pandas as pd 

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


if __name__=="__main__":
    root_path='/git/datasets/beigang_data/'
    #data_path='all_variables_for_mine_price.csv'
    #data_path_new='all_variables_for_mine_price_merged.csv'
    data_path="runmin_an_factors.csv"
    all_targets=["4500K1.0S",'5000K0.8S','5500K0.8S']
    target=all_targets[2]
    if target in all_targets:
        choose_variable(root_path,data_path,all_targets,target)
    elif target=="merged":
        merge_target(root_path,data_path,target)

      





