
#python infer.py --weights_path Checkpoint/20241030-ct2.pth --Input_path Input/Test_case_Lung --Output_path Output \
#      --Config_path Configs/config_lung.yaml

#python infer.py --weights_path Checkpoint/20241030-ct2.pth --Input_path Input/Test_case_Abdomen --Output_path Output \
#      --Config_path Configs/config_abdomen.yaml

python infer.py --weights_path Checkpoint/20241030-ct2.pth --Input_path Input/Test_case_Head --Output_path Output \
      --Config_path Configs/config_head.yaml
