export PATH=/usr/local/cuda-10.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH}
export AUTOGRAPH_VERBOSITY=10
export CUDA_VISIBLE_DEVICES=4,1
export TF_CPP_MIN_LOG_LEVEL='3'
export FLAGS_fraction_of_gpu_memory_to_use=0.92
export FLAGS_allocator_strategy='autogrowth'
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
--loss_matching_G 0.01 \
--loss_matching_D 1 \
--loss_sim 1 
# --dataset vggface \
# --image_width 28 \
# --batch_size 10  \
# --experiment_title vggface1way3shot  \
# --selected_classes 1 \
# --support_number 3  \
# --loss_G 1 \
# --loss_D 1 \
# --loss_CLA 1 \
# --loss_recons_B 0.1 \
# --loss_matching_D 1 \