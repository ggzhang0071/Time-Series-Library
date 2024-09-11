#!/bin/bash

# 获取当前时间戳作为日志文件名
timestamp=`date +%Y%m%d%H%M%S`
logfile="Logs/${timestamp}.log"

# 删除旧的日志文件
rm Logs/*.log 
#rm test_results/* -rf 
rm runs/* -rf 


#bash scripts/beigang_script/iTransformer.sh 63  2>&1 | tee -a "${logfile}"

#bash  scripts/beigang_script/DLinear.sh  45  2>&1 | tee -a "${logfile}"

#bash  scripts/beigang_script/PatchTST.sh   2>&1 | tee -a "${logfile}"

bash scripts/stock/PatchTST.sh 2>&1 | tee -a "${logfile}"



