from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,visual_prediction
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from utils.losses import mape_loss,mape1_loss,mase_loss, smape_loss, r2_loss
from utils.tools import reconstruct_series_from_preds
from utils.tools import diff_batch
from utils.metrics import  calculate_accuracy
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings('ignore')

def _select_criterion(loss_name='MSE'):
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAPE':
        return mape_loss()
    elif loss_name == 'MAPE1':
        return mape1_loss()
    elif loss_name == 'MASE':
        return mase_loss()
    elif loss_name == 'SMAPE':
        return smape_loss()
    elif loss_name=="r2":
        return r2_loss()

print("Loading Exp_Long_Term_Forecast module...")

class Exp_Long_Term_Forecast(Exp_Basic):

    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.writer = SummaryWriter(log_dir=f"runs/{args.model}_{args.task_name}_pred_len_{args.pred_len}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,_,_) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if self.args.loss=="r2":
                    loss=criterion(None,1,outputs,batch_y)
                elif self.args.loss=="MSE":
                    loss = criterion(outputs, batch_y)


                total_loss.append(loss)
        if len(total_loss) > 0:
            total_loss = np.average(total_loss[0].cpu().numpy())
        else:
            # 如果 total_loss 为空，返回 np.nan
            total_loss = np.nan  
            print("Warning: total_loss is empty. Returning default loss value.")

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print(f"train size:{train_data[0][0].shape}, val size: {vali_data[0][0].shape} test size: {test_data[0][0].shape}")


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer() 
        #criterion = self._select_criterion()
        criterion = _select_criterion(self.args.loss)


        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,_,_) in enumerate(train_loader):     
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        if self.args.loss=="r2":
                            loss=criterion(None,1,outputs,batch_y)
                        elif self.args.loss=="MSE":
                            loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.loss=="r2":
                        loss=criterion(None,1,outputs,batch_y)
                    elif self.args.loss=="MSE":
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")

                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            # Add histogram for the parameter gradients if they exist
                            self.writer.add_histogram(f'{name}.grad', param.grad, epoch)
                        else:
                            print(f'No gradient for {name} at epoch {epoch}')

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss}, Vali Loss: {vali_loss}, Test Loss: {test_loss}")


            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/vali', vali_loss, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
        

            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.writer.close()

        return self.model, vali_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_original_y,batch_origial_stamp) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                batch_original_y=batch_original_y[:, -self.args.pred_len:].unsqueeze(-1).to(self.device)
                batch_origial_stamp=batch_origial_stamp[:, -self.args.pred_len:].unsqueeze(-1).to(self.device)
          
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_original_y=batch_original_y.detach().cpu().numpy()
                batch_origial_stamp=batch_origial_stamp.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    # 预处理outputs，使其适合inverse_transform方法
                    if outputs.ndim == 3 and outputs.shape[0] == 1:
                        outputs = outputs.squeeze(0)  # 压缩第0维，如果是(batch_size, pred_len, features)且batch_size为1
                    # 应用inverse_transform并重新塑形
                    if outputs.ndim <= 2:
                        outputs = test_data.inverse_transform(outputs).reshape(shape)
                    else:
                        # 如果outputs是三维且无法压缩为二维，逐个处理每个样本
                        processed_outputs = []
                        for ii in range(outputs.shape[0]):
                            processed_sample = test_data.inverse_transform(outputs[ii]).reshape(outputs[ii].shape)
                            len(test_data[0][0])
                            processed_outputs.append(processed_sample)
                        outputs = np.array(processed_outputs).reshape(shape)
                    
                    # 应用相同的逻辑于batch_y
                    if batch_y.ndim == 3 and batch_y.shape[0] == 1:
                        batch_y = batch_y.squeeze(0)
                    if batch_y.ndim <= 2:
                        batch_y = test_data.inverse_transform(batch_y).reshape(shape)
                    else:
                        processed_batch_y = []
                        for kk in range(batch_y.shape[0]):
                            processed_sample = test_data.inverse_transform(batch_y[kk]).reshape(batch_y[kk].shape)
                            processed_batch_y.append(processed_sample)
                        batch_y = np.array(processed_batch_y).reshape(shape)

                        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i >=0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape  # e.g., (batch_size, seq_length, feature_dim)
                        # 将 input 展平为二维数组
                        input_reshaped = input.reshape(-1, shape[-1])  # shape: (batch_size * seq_length, feature_dim)
                        # 逆变换
                        input_transformed = test_data.inverse_transform(input_reshaped)
                        # 将 input 恢复到原始形状
                        input = input_transformed.reshape(shape)
                    
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd1 = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, self.args.pred_len, pd1, os.path.join(folder_path,  str(i) + '.png'))


        if not preds:  
            print("Preds is an empty list, the trial is failed")  
        else:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
            # dtw calculation
            if self.args.use_dtw:
                dtw_list = []
                manhattan_distance = lambda x, y: np.abs(x - y)
                for i in range(preds.shape[0]):
                    x = preds[i].reshape(-1,1)
                    y = trues[i].reshape(-1,1)
                    if i % 100 == 0:
                        print("calculating dtw iter:", i)
                    d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                    dtw_list.append(d)
                dtw = np.array(dtw_list).mean()
            else:
                dtw = -999 
            #len(preds) preds[0].shape  len(trues) trues[0].shape len(batch_original_y) batch_original_y[0].shape
            if self.args.target_preprocess=="diff" and self.args.inverse==True:
                """#测试做差分的数据和原来的数据是不是一样
                diff_batch_y=diff_batch(batch_original_y[0,:,-1])
                print(trues[0,:,-1]-diff_batch_y)"""
               
                preds=reconstruct_series_from_preds(preds,batch_original_y)
                trues=batch_original_y
            
            mae, mse, rmse, mape, mspe, r2= metric(preds, trues)
            Min_acc,Max_acc=calculate_accuracy(preds[1:],trues[1:])
            print(f'mse:{mse}, rmse:{rmse[0]}, mae:{mae}, mape:{mape}, r2:{r2}, Min_acc:{Min_acc}, Max_acc:{Max_acc}, dtw:{dtw}')

            if Min_acc>(1/(self.args.pred_len-1))*1.5 and Max_acc>(1/(self.args.pred_len-1))*1.5:
                for  jj in range(trues.shape[0]):
                    visual_prediction(trues[jj], preds[jj],os.path.join(folder_path, f"pred_len_{self.args.pred_len}_minacc_{Min_acc}_maxacc_{Max_acc}_{kk}.png"))

                
                if test_data.scale and self.args.inverse:
                    f = open("result_long_term_forecast_inverse.txt", 'a')
                else:
                    f = open("result_long_term_forecast.txt", 'a')

                f.write(setting + "  \n")
                f.write(f'mse:{mse},  rmse:{rmse[0]}, mae:{mae}, mape:{mape}, r2:{r2}, Min_acc:{Min_acc}, Max_acc:{Max_acc}, dtw:{dtw}')
                f.write('\n')
                f.write('\n')
                f.close()

                #保存成 numpy 数据格式 或者csv 格式
                if self.args.save_format=="npy":
                    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,Min_acc,Max_acc]))
                    np.save(folder_path + 'pred.npy', preds)
                    np.save(folder_path + 'true.npy', trues)
                    np.save(folder_path + 'time_stamp.npy',batch_origial_stamp)
                elif self.args.save_format=="csv":
                    metrics = {
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'mape': mape,
                        'mspe': mspe,
                        'Min_acc': Min_acc,
                        'Max_acc': Max_acc
                    }
                    df = pd.DataFrame([metrics])
                    df.to_csv(folder_path + f"metrics_{self.args.target}_pred_len_{self.args.pred_len}.csv", index=False)
                    preds_array = np.array(preds).squeeze()  # Squeeze to remove single-dimensional entries
                    np.savetxt(folder_path + f"pred_{self.args.target}_pred_len_{self.args.pred_len}.csv", preds_array, delimiter=',', fmt='%d')

                    trues_array = np.array(trues).squeeze()  # Squeeze to remove single-dimensional entries
                    np.savetxt(folder_path + f"true_{self.args.target}_pred_len_{self.args.pred_len}.csv", trues_array, delimiter=',', fmt='%d')

                    batch_origial_stamp_np = batch_origial_stamp.squeeze()  # Remove the last dimension, resulting in shape (99, 15)

                    # Convert NumPy array to DataFrame
                    batch_origial_stamp_pd = pd.DataFrame(batch_origial_stamp_np)
                    batch_origial_stamp_pd = batch_origial_stamp_pd.apply(pd.to_datetime, unit='s')

                    batch_origial_stamp_pd = batch_origial_stamp_pd.applymap(lambda x: x.strftime('%Y-%m-%d'))
                    batch_origial_stamp_pd.to_csv(folder_path + f"time_stamp_{self.args.target}_pred_len_{self.args.pred_len}.csv", index=False, header=False)

                    print(f"Timestamp, true values, and prediction values was saved to folder {folder_path}")
        return  


