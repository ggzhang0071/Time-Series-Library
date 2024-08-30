export CUDA_VISIBLE_DEVICES=2


# 导入参数
source optuna_best_params.sh
model_name=PatchTST
task_name="long_term_forecast"

# 检查是否传递了参数组选择
if [[ "$1" =~ ^[0-9]+$ ]]; then
    "optuna_params_$1"
    config_path=""
    num_trial=1
    train_epochs=10
else
    echo "Using the optuna for hypterparameter searing"
    e_layers=5 
    learning_rate=0.007 
    batch_size=120
    train_epochs=10
    config_path="./scripts/beigang_script/param_config_${task_name}_${model_name}.json"
    num_trial=100
    pred_len=20
    #all_targets=['5000K0.8S','5500K0.8S',"4500K1.0S"]
    target=5000K0.8S
fi 


root_path="/git/Time-Series-Library/beigang_data"
#data_path="all_variables_for_mine_price_${target}.csv"
#data_path="runmin_an_factors_${target}.csv"
data_path="early_variables_for_mine_price_${target}.csv"

seq_len=96

for d_model in 120; 
do 
  for pred_len in $pred_len;
   do 
    nohup python   run_optuna.py \
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
      --e_layers $e_layers \
      --d_layers 1 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --batch_size $batch_size \
      --freq 'd' \
      --train_epochs $train_epochs \
      --d_model $d_model \
      --d_ff $((d_model * 2)) \
      --target_preprocess "diff" \
      --learning_rate $learning_rate \
      --itr 1 \
      --patience 10 \
      --inverse \
      --target $target \
      --config "$config_path" \
      --augmentation_ratio 1 \
      --num_trial $num_trial \
      --save_format "csv"  \
      --itr 1
  done
done