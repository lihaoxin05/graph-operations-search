export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PYTHONPATH:/path/to/faster-rcnn.pytorch/lib/model
cd ../..
python main.py \
--video_path /path/to/20bn-something-something-v1/frames \
--proposal_path /path/to/Something-Something-V1/region_proposal \
--annotation_path /path/to/Something-Something-V1/lists \
--result_path /path/to/experiments \
--dataset SomethingSomethingV1 --n_classes 174 \
--learning_rate 0.01 --weight_decay 1e-4 --lr_patience 10 \
--model_depth 50  --basenet_fixed_layers 4 --n_box_per_frame 10 \
--step_per_layer 3 --arch_learning_rate 1e-4 --arch_weight_decay 1e-3 --op_loss_weight 1e-1 \
--batch_size 64 --n_epochs 30 --n_threads 8 \
--pretrain_path /path/to/pretrained_backbone.pth 
