export CUDA_VISIBLE_DEVICES=3

target="4500K1.0S" 

root_path="/git/datasets/beigang_data/${target}"
#data_path='all_variables_for_mine_price_4500K1.0S.csv'
data_path="runmin_an_factors_4500K1.0S.csv"
#data_path="early_variables_for_mine_price.csv"

enc_in_choice=78
data_path="runmin_an_factors_${target}.csv"

d_model=14
model_name=iTransformer 
seq_len=756
task_name="long_term_forecast"

config_path="./scripts/beigang_script/optuna_opt/param_config_${task_name}_${model_name}.json"

for pred_len in 3 5 7 9 11 13 15
do 
# run_optuna.py  run.py  
nohup python   run_optuna.py \
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
  --e_layers 5 \
  --d_layers 4 \
  --factor 3 \
  --enc_in $enc_in_choice \
  --dec_in $enc_in_choice \
  --c_out 1 \
  --des 'Exp' \
  --train_epochs 20 \
  --dropout 0.1 \
  --d_model $d_model \
  --d_ff $((d_model*4)) \
  --p_hidden_dims  16 16 \
  --target_preprocess  "diff" \
  --batch_size 128 \
  --learning_rate 0.01 \
  --itr 1   \
  --patience 10 \
  --target $target \
  --inverse \
  --config ${config_path} \
  --loss 'MSE'  
done 

for pred_len in 3 5 7 9 11 13 15
do 
# run_optuna.py  run.py  
python -m pdb  /git/Time-Series-Library/run_optuna.py \
  --task_name  $task_name \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --target_preprocess "diff" \
  --model_id   beigang_756_60 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 765 \
  --label_len 60 \
  --pred_len $pred_len \
  --e_layers 5 \
  --d_layers 4 \
  --factor 3 \
  --output_task $output_task \
  --enc_in $enc_in_choice \
  --dec_in $enc_in_choice \
  --c_out 1 \
  --des 'Exp' \
  --train_epochs 20 \
  --dropout 0.1 \
  --d_model $d_model \
  --d_ff $((d_model*4)) \
  --p_hidden_dims  16 16 \
  --batch_size 128 \
  --learning_rate 0.01 \
  --itr 1   \
  --patience 10 \
  --target $target \
  --inverse \
  --config ${config_path} \
  --loss 'MSE'  
done 

