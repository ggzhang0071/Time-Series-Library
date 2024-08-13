#!/bin/bash

# 获取当前时间戳作为日志文件名
timestamp=`date +%Y%m%d%H%M%S`
logfile="Logs/${timestamp}.log"

# 删除旧的日志文件
#rm Logs/*.log

for pred_len in 

bash scripts/beigang_script/optuna_opt/iTransformer_long_term_forecast.sh 1 2>&1 | tee -a "${logfile}"
