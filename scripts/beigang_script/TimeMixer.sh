#export CUDA_VISIBLE_DEVICES=0


model_name=TimeMixer
task_name=long_term_forecast

target="4500K1.0S" 
root_path="/git/datasets/beigang_data/${target}"
#data_path='all_variables_for_mine_price_4500K1.0S.csv'
data_path="runmin_an_factors_${target}.csv"
#data_path="early_variables_for_mine_price.csv"


seq_len=96
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
batch_size=16
train_epochs=20
patience=10
enc_in_choice=78


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
    config_path="./scripts/beigang_script/param_config_${task_name}_${model_name}.json"
    num_trial=5
fi
for pred_len in 5
do 
python  run_optuna.py \
  --task_name $task_name \
  --is_training 1 \
  --root_path $root_path\
  --data_path $data_path \
  --model_id "beigang_${seq_len}_${pred_len}" \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in_choice \
  --dec_in $enc_in_choice \
  --c_out $enc_in_choice \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --inverse \
  --target $target \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --config "${config_path}" \
  --num_trial $num_trial \
  --itr 1 
  
done
./scripts/beigang_script/param_config_long_term_forecast_TimeMixer.json


