#img="nvcr.io/nvidia/pytorch:22.06-py3" 
#img="pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime"
img="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel"

docker run --gpus all  --privileged=true   --workdir /git --name "tsl"  -e DISPLAY --ipc=host -d --rm  -p 6334:8889 \
-v /home/ggzhang/Time-Series-Library:/git/Time-Series-Library \
-v /home/ggzhang/datasets:/git/datasets \
$img sleep infinity

docker exec -it tsl /bin/bash  

#docker images  |grep "pytorch"  |grep "21."

#docker stop  tsl 


