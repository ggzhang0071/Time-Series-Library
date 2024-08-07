export CUDA_VISIBLE_DEVICES=0

target="4500K1.0S" 

root_path="/git/datasets/beigang_data/${target}"
#data_path='all_variables_for_mine_price_4500K1.0S.csv'
data_path="runmin_an_factors_4500K1.0S.csv"
#data_path="early_variables_for_mine_price.csv"

enc_in_choice=78
data_path="runmin_an_factors_${target}.csv"

d_model=14
model_name=iTransformer 

task_name="long_term_forecast"
output_task="classification"

config_path="/git/Time-Series-Library/scripts/beigang_script/optuna_opt/param_config_${task_name}.json"
config_path="/git/Time-Series-Library/scripts/beigang_script/optuna_opt/param_config_long_term_forecast.json"


# run_optuna.py  run.py  
python   /git/Time-Series-Library/run_optuna.py \
  --task_name  $task_name \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --target_preprocess "target_classification" \
  --model_id   beigang_756_60 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 765 \
  --label_len 60 \
  --pred_len 6 \
  --e_layers 5 \
  --d_layers 4 \
  --factor 3 \
  --output_task $output_task \
  --enc_in $enc_in_choice \
  --dec_in $enc_in_choice \
  --c_out 1 \
  --des 'Exp' \
  --train_epochs 10 \
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
  --loss 'CrossEntropy'  

<<COMMENT
python   /git/Time-Series-Library/run_optuna.py \
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
