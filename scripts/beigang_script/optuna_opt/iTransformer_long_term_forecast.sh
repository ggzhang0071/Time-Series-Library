export CUDA_VISIBLE_DEVICES=3

target="4500K1.0S" 
root_path="/git/datasets/beigang_data/${target}"
#data_path='all_variables_for_mine_price_4500K1.0S.csv'
data_path="runmin_an_factors_${target}.csv"
#data_path="early_variables_for_mine_price.csv"

model_name=iTransformer   
enc_in_choice=78
seq_len=756
task_name="long_term_forecast"


# 导入参数
source optuna_best_params.sh

# 检查是否传递了参数组选择
if [ "$1" == "1" ]; then
    optuna_params_1
    config_path=""
elif [ "$1" == "2" ]; then
    optuna_params_2
    config_path=""
else
    echo "Using the optuna for hypterparameter searing"
    d_model=14
    e_layers=5
    learning_rate=0.007
    batch_size=120
    config_path="./scripts/beigang_script/optuna_opt/param_config_${task_name}_${model_name}.json"
fi



#3 5 7 9 11 13 15
for pred_len in 13
do 
# run_optuna.py  run.py  
 python   run_optuna.py \
  --task_name  $task_name \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id  "beigang_${seq_len}_${pred_len}" \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len $seq_len \
  --label_len 60 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 4 \
  --factor 3 \
  --enc_in $enc_in_choice \
  --dec_in $enc_in_choice \
  --c_out 1 \
  --des 'Exp' \
  --freq 'd'\
  --train_epochs 10 \
  --dropout 0.1 \
  --d_model $d_model \
  --d_ff $((d_model*4)) \
  --p_hidden_dims  16 16 \
  --target_preprocess  "diff" \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --itr 1   \
  --patience 10 \
  --inverse \
  --target $target \
  --config "$config_path" \
  --loss 'MSE'  
done 
