import os 
import pandas as pd
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler




class ParquetDataset(Dataset):
    def __init__(self, args,root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, target_preprocess=None,timeenc=0, freq='h',  seasonal_patterns=None,augmentation_ratio=0,file_extension="parquet", engine="pyarrow"):
        # size [seq_len, label_len, pred_len]
        if size is  None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2] 
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.args=args 
        self.features = features
        self.target = target
        self.target_preprocess=target_preprocess
        self.augmentation_ratio=augmentation_ratio
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag=flag 
        self.file_extension = file_extension 
        self.engine=engine

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    def read_single_file(self, file_path):
        df = pd.read_parquet(file_path, columns=['datetime', 'score'], engine=self.engine)
        df['date'] = pd.to_datetime(df['datetime']).dt.date
        df.drop(columns=['datetime'], inplace=True)
        return df

    def __read_data__(self):
        self.file_paths = [os.path.join(self.root_path, filename) 
                           for filename in os.listdir(self.root_path) 
                           if filename.endswith(self.file_extension)]
        df_raw0=[]
        counter=0
        for file_path in self.file_paths:
            counter+=1
            print(counter)
            if counter<=5: 
                # 读取每个文件中 'score' 列的数据
                df = pd.read_parquet(file_path, engine='pyarrow')  # or engine='fastparquet'
                df_selected = df[['datetime', 'score']]
                # Replace 'datetime' column with a new 'date' column
                df_selected['date'] = pd.to_datetime(df_selected['datetime']).dt.date
                
                # Drop the original 'datetime' column
                df_selected = df_selected.drop(columns=['datetime'])
                
                # Append the DataFrame to the list
                df_raw0.append(df_selected)

            # Concatenate all DataFrames based on 'date'
            df_raw = pd.concat(df_raw0).sort_values(by='date')

            # Reset the index (optional)
            df_raw.reset_index(drop=True, inplace=True)
         

    
    
        if self.target_preprocess=="diff" and self.target!="":
            y=df_raw[self.target]
            y_diff=y.shift(-1)/y -1 
            df_raw[self.target]=y_diff

        #重新排列，以便数据后续更好的特定剔除target 和date 项目
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols +[self.target]]
       
        
        
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 是否对数据做尺度化
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
                    
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 加入时间戳信息作为feature
        df_stamp = df_raw[['date']][border1:border2]
        original_stamp=df_stamp.copy(deep=True)
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 截取train val and test 
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        if self.target_preprocess=="diff" and self.flag=="test":
            # 这个orginal——target 才有用，如果不做差分diff，做scale 倒是不一定
            self.test_original_target=y[border1:border2]
            self.original_stamp=original_stamp


        if self.set_type == 0 and self.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index 
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]


        if self.target_preprocess=="diff" and self.flag=="test" and self.scale:
            test_original_target=self.test_original_target.values[r_begin:r_end]
            """#测试做差分之后的数据和原来的数据是不是相同的
            y_shift=diff_batch(target_original)
            print(y_shift[:5],self.scaler.inverse_transform(seq_y[:5])[:,-1])"""
            original_stamp=self.original_stamp[r_begin:r_end]['date']
            original_stamp = pd.to_datetime(original_stamp)
            timestamps_int = original_stamp.astype('int64') // 10**9
            original_stamp_unix = timestamps_int.to_numpy()
        else:
            test_original_target=np.zeros((r_end-r_begin,))
            original_stamp_unix=np.zeros((r_end-r_begin,))
            

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, test_original_target, original_stamp_unix

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
