# SubtleCT DNE Inference

##  1. Prepare Environment
Install dependencies:
Only a few core packages are required, including pydicom, SimpleITK, and PyTorch:

```
pydicom==2.4.4
SimpleITK==2.3.1
torch==2.3.1
```

Alternatively, you can build the environment directly using the provided [Dockerfile](https://github.com/ZhangZhiHao233/subtlect-inference/blob/main/Dockerfile)

Steps to build the final image based on subtle/base_cu12_ubuntu20_py310:latest using the Dockerfile.

```
Step 1. Place the code folder subtlect-inference and the Dockerfile in the same directory
Step 2. Run: docker build -t subtlect_infer:v2 .
Step 3. Run the container with: docker run --gpus device=0 -it subtlect_infer:v2
Step 4. (Inside the container)
	4.1 cd subtlect-inference
	4.2 bash run.sh
```
##  2. Prepare Files
Directory structure:
```
.
├─ Input/                 # DICOM series to process
├─ Output/                # Output folder
├─ Checkpoint/            # Model weights (.pth)
├─ Configs				  # Config files
	├─ config_lung.yaml
	├─ config_abdomen.yaml
	└─ config_head.yaml    
├─ log   
```
Test cases and model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/12d-gMxLfGVXxOInf_F_YIkzrq7_vjqof?usp=sharing).

##  3. Run Inference
You can either run directly with Python:
```
python infer.py \
  --weights_path checkpoint/20241030-ct2.pth \
  --Input_path Input \
  --Output_path Output \
  --Config_path config_head.yaml
```
Or simply run the shell script:
```
bash run.sh
```

##  4. Output
Denoised DICOM series will be saved under Output/ with updated SeriesDescription tags.
Logs are written to log/inference.log.