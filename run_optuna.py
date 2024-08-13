import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import optuna 
import time
import logging
import json
import pandas as pd 
logging.basicConfig(level=logging.INFO)

def update_args_(args, params):
  """updates args in-place"""
  dargs = vars(args)
  dargs.update(params)

def create_main():
    def main(trial):
        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        parser = argparse.ArgumentParser(description='TimesNet')
        # basic config
        parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
        parser.add_argument('--model', type=str, required=True, default='Autoformer',
                            help='model name, options: [Autoformer, Transformer, TimesNet]')

        # data loader
        parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
        parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
        parser.add_argument('--target_preprocess', type=str, default="",help='preprocess for target')
        parser.add_argument('--target', type=str, default="4500K1.0S", help='target feature in S or MS task')
        parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

        # forecasting task
        parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

        # inputation task
        parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

        # anomaly detection task
        parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

        # model define
        parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
        parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
        parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
        parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=1, help='attn factor')
        parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                            default=True)
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_task', type=str, default='', help='regression or classification')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        parser.add_argument('--channel_independence', type=int, default=1,
                            help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                            help='method of series decompsition, only support moving_avg or dft_decomp')
        parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
        parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
        parser.add_argument('--down_sampling_method', type=str, default=None,
                            help='down sampling method, only support avg, max, conv')
        parser.add_argument('--seg_len', type=int, default=48,
                            help='the length of segmen-wise iteration of SegRNN')

        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
        parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
        parser.add_argument('--des', type=str, default='test', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)


        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # de-stationary projector params
        parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                            help='hidden layer dimensions of projector (List)')
        parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

        # metrics (dtw)
        parser.add_argument('--use_dtw', type=bool, default=False, 
                            help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
        
        # Augmentation
        parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
        parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
        parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
        parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
        parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
        parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
        parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
        parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
        parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
        parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
        parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
        parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
        parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
        parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
        parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
        parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
        parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
        parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
        parser.add_argument('--config', type=str, required=False, default="", help='Path to config file')

        parser.add_argument('--num_trial', type=int, default=2, help="optuna trial numbers")

        args = parser.parse_args()
        if args.config!="":
            with open(args.config, 'r') as fid:
                param_config=json.load(fid)
            if trial is not  None and param_config!=None:
                for param_name, attributes in param_config.items():
                    param_type = attributes['type']
                    if param_type == 'float':
                        # 更新 args 中对应的属性
                        setattr(args, param_name, trial.suggest_float(param_name, attributes['low'], attributes['high']))
                    elif param_type == 'int':
                        setattr(args, param_name, trial.suggest_int(param_name, attributes['low'], attributes['high']))
                    # 如果有其他类型，可以在这里添加处理逻辑

        # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        args.use_gpu = True if torch.cuda.is_available() else False

        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]  
        else:
            pass
            #args.device_ids= [args.devices]
            #args.gpu =args.devices
            #args.gpu  = trial.suggest_categorical('gpu_id', [0, 1, 2, 3])  # 假设有 4 个 GPU

        """if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model"""


        print('Args in experiment:')
        print_args(args)

        if args.task_name == 'long_term_forecast':
            Exp = Exp_Long_Term_Forecast
        elif args.task_name == 'short_term_forecast':
            Exp = Exp_Short_Term_Forecast
        elif args.task_name == 'imputation':
            Exp = Exp_Imputation
        elif args.task_name == 'anomaly_detection':
            Exp = Exp_Anomaly_Detection
        elif args.task_name == 'classification':
            Exp = Exp_Classification
        else:
            Exp = Exp_Long_Term_Forecast
      

        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                exp = Exp(args)  # set experiments
                setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}\
                 _ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}\
                 _dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}\
                 _df{args.d_ff}_expand{args.expand}_dc{args.d_conv}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}"


                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                model, vali_loss =exp.train(setting)
                


                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}\
                _sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}\
                    _el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_expand{args.expand}_dc{args.d_conv}\
                        _fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}"


            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
        return vali_loss
    return main
if __name__=="__main__":
    n_trials=100
    start_time=time.time()
    study = optuna.create_study(direction='minimize')
    objective =create_main()
    study.optimize(objective, n_trials=n_trials)
    end_time=time.time()
    optuna_time=end_time-start_time  

    print("Optuna best params: ", study.best_params)
    print("Optuna best value: ", study.best_value)  # 最优值
    print("Optuna runtime: ", optuna_time, "sec")

    # CSV 文件路径
    file_path = 'optuna_best_params.csv'

    # 将最优参数和最优值保存到 CSV 文件中
    best_params = study.best_params
    best_params['best_value'] = study.best_value  # 添加最优值到字典中
    best_params["n_trials"]=n_trials
    df = pd.DataFrame([best_params])

    # 检查文件是否存在，如果存在则追加，否则创建新文件
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)  




 

