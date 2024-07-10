export CUDA_VISIBLE_DEVICES=0


root_path='/git/datasets/beigang_data/'
#data_path='all_variables_for_mine_price_4500K1.0S.csv'
data_path="runmin_an_factors_4500K1.0S.csv"
#data_path="early_variables_for_mine_price.csv"

# 检查 data_path 并设置模型参数
if [ "$data_path" == "all_variables_for_mine_price_4500K1.0S.csv" ]; then
    enc_in_choice=86
    target="4500K1.0S"
elif [ "$data_path" == "runmin_an_factors_4500K1.0S.csv" ]; then
    enc_in_choice=78
    target="4500K1.0S"
elif [ "$data_path" == "runmin_an_factors_5000K0.8S.csv" ]; then
    enc_in_choice=78
    target="5000K0.8S"
elif [ "$data_path" == "runmin_an_factors_5500K0.8S.csv" ]; then
    enc_in_choice=78
    target="5500K0.8S"
else  
    echo "未识别的 data_path: $data_path"
    exit 1
fi

d_model=64
model_name=iTransformer  

# run_optuna.py  run.py
 python -m pdb    /git/Time-Series-Library/run_optuna.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id   beigang_756_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 765 \
  --label_len 60 \
  --pred_len 5 \
  --e_layers 4 \
  --d_layers 4 \
  --factor 3 \
  --enc_in $enc_in_choice \
  --dec_in $enc_in_choice \
  --c_out 1 \
  --des 'Exp' \
  --dropout 0.1 \
  --d_model $d_model \
  --d_ff $((d_model*4)) \
  --p_hidden_dims  16 16 \
  --batch_size 512 \
  --train_epochs 30 \
  --learning_rate 0.01 \
  --itr 50   \
  --target $target \
  --loss 'SMAPE'

<<COMMENT
python   -m pdb  /git/Time-Series-Library/run_optuna.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --data "custom" \
  --seasonal_patterns 'Monthly' \
  --model_id beigang \
  --model $model_name \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --target 4500K1.0S \
  --loss 'SMAPE'
COMMENT
