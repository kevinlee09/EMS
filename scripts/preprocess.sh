input_data="../example_data/"
gpu_id="0"

cd preprocess
python face_recon.py --input ${input_data}  --gpu_ids ${gpu_id}
python face_nicp.py --input ${input_data} --gpu_ids ${gpu_id} 
python process_img_cam.py --input ${input_data} --gpu_ids ${gpu_id} 