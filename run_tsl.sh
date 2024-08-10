#!/bin/bash

# 获取当前时间戳作为日志文件名
timestamp=`date +%Y%m%d%H%M%S`
logfile="Logs/${timestamp}.log"

# 删除旧的日志文件
#rm Logs/*.log

<<COMMENT
bash ./scripts/classification/DLinear.sh   2>&1 | tee -a "${logfile}"
bash ./scripts/classification/FiLM.sh    2>&1 | tee -a "${logfile}"
bash ./scripts/classification/MICN.sh    2>&1 | tee -a "${logfile}"
bash ./scripts/classification/Reformer.sh   2>&1 | tee -a "${logfile}"
bash ./scripts/classification/iTransformer.sh   2>&1 | tee -a "${logfile}"
bash ./scripts/classification/Crossformer.sh  2>&1 | tee -a "${logfile}"
bash ./scripts/classification/FEDformer.sh   2>&1 | tee -a "${logfile}"
bash ./scripts/classification/LightTS.sh    2>&1 | tee -a "${logfile}"
bash ./scripts/classification/Pyraformer.sh   2>&1 | tee -a "${logfile}"
bash ./scripts/classification/Transformer.sh   2>&1 | tee -a "${logfile}"

# long-term forecast
#bash ./scripts/short_term_forecast/iTransformer_M4.sh
#bash ./scripts/long_term_forecast/Weather_script/iTransformer.sh
bash ./scripts/long_term_forecast/Traffic_script/iTransformer.sh   2>&1 | tee -a "${logfile}"
bash ./scripts/long_term_forecast/ETT_script/iTransformer_ETTh2.sh   2>&1 | tee -a "${logfile}"
bash ./scripts/long_term_forecast/ECL_script/iTransformer.sh   2>&1 | tee -a "${logfile}"
bash ./scripts/anomaly_detection/MSL/iTransformer.sh   2>&1 | tee -a "${logfile}"
COMMENT

bash scripts/beigang_script/optuna_opt/iTransformer_long_term_forecast.sh   2>&1 | tee -a "${logfile}"
#bash scripts/beigang_script/optuna_opt/TimesNet.sh     2>&1 | tee -a "${logfile}"

#bash ./scripts/classification/DLinear.sh   2>&1 | tee -a "${logfile}"

#bash scripts/beigang_script/optuna_opt/TimesNet_classification.sh   2>&1 | tee -a "${logfile}"

#bash scripts/beigang_script/optuna_opt/iTransformer_classification.sh 2>&1 | tee -a "${logfile}"











