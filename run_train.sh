export PATH=/usr/local/cuda-10.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH}
export AUTOGRAPH_VERBOSITY=10
export CUDA_VISIBLE_DEVICES=2
export TF_CPP_MIN_LOG_LEVEL='3'
export FLAGS_fraction_of_gpu_memory_to_use=0.92
export FLAGS_allocator_strategy='autogrowth'

# 输入进来的图片路径，格式需要是npy
data_root="/model/huzhe/Dataset/vgg_face_data.npy" 

# 输出的log，可视化图像，保存的模型的根路径。注意experiment_title会在这个路径下的基础上创建目录
output_root="/model/huzhe/memoryGAN"

python train_dagan_with_matchingclassifier.py \
--dataset vggface \
--image_width 96 \
--batch_size 20  \
--experiment_title vggface1way3shot_all \
--selected_classes 1 \
--support_number 3  \
--loss_G 1 \
--loss_D 1 \
--loss_CLA 1 \
--loss_recons_B 1 \
--loss_matching_G 1 \
--loss_matching_D 1 \
--loss_sim 1 \
--data_root ${data_root} \
--output_root ${output_root}
