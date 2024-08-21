export CUDA_VISIBLE_DEVICES=3
model_name=TimesNet

target="4500K1.0S" 

root_path="/git/datasets/beigang_data/${target}"
#data_path='all_variables_for_mine_price_4500K1.0S.csv'
data_path="runmin_an_factors_4500K1.0S.csv"
#data_path="early_variables_for_mine_price.csv"

enc_in_choice=78

task_name=long_term_forecast
seq_len=96 


# 导入参数
source optuna_best_params.sh

# 检查是否传递了参数组选择
if [[ "$1" =~ ^[0-9]+$ ]]; then
    "optuna_params_$1"
    config_path=""
    num_trial=1
    train_epochs=20
else
    echo "Using the optuna for hypterparameter searing"
    d_model=128  
    #d_mode=256
    e_layers=5 
    learning_rate=0.007 
    batch_size=120
    train_epochs=10
    config_path="./scripts/beigang_script/optuna_opt/param_config_${task_name}_${model_name}.json"
    num_trial=5

fi

for pred_len in 20  15 10 5 
do 
python  run_optuna.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id "beigang_${seq_len}_${pred_len}" \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in_choice \
  --dec_in $enc_in_choice \
  --c_out $enc_in_choice \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --des 'Exp' \
  --target_preprocess 'diff' \
  --target $target \
  --inverse \
  --itr 1
done 
