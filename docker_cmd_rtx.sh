#img="nvcr.io/nvidia/pytorch:22.06-py3" 
#img="pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime"
img="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel"

docker run --gpus all  --privileged=true   --workdir /git --name "tsl"  -e DISPLAY --ipc=host -d --rm  -p 6334:8889 \
-e LANG=C.UTF-8 -e LANGUAGE=C.UTF-8 -e LC_ALL=C.UTF-8 \
-v /home/ggzhang/Time-Series-Library:/git/Time-Series-Library \
-v /home/ggzhang/datasets:/git/datasets \
-v  /home/ruiminan/model_code/beigang_price_prediction/factor_processor/factor_preselect_bymodel/analysis_notebook:/git/dataset1 \
$img sleep infinity

docker exec -it tsl /bin/bash  

#docker images  |grep "pytorch"  |grep "21."

#docker stop  tsl 


