python -m torch.distributed.launch main_resnet50.py \
--dataset imagenet \
--data /data3/imagenet-data/raw-data \
--arch resnet50_official \
--depth 50 \
--lr 0.1 \
--epochs 90 \
--schedule 30 60 \
--batch-size 256 \
--test-batch-size 64 \
--save ./EBTrain-ImageNet/ResNet50 \
--momentum 0.9 \
--sparsity-regularization \
--gpu_ids 0,1,2,3


python -m torch.distributed.launch main_resnet50.py \
--dataset imagenet \
--data /ssd1/bansa01/imagenet_final \
--arch resnet50_official \
--depth 50 \
--lr 0.1 \
--epochs 90 \
--schedule 30 60 \
--batch-size 256 \
--test-batch-size 64 \
--save ./EBTrain-ImageNet/ResNet50 \
--momentum 0.9 \
--sparsity-regularization \
--gpu_ids 0,1,2,3 \
--resume ./EBTrain-ImageNet_Moco/ResNet50/ckpt0.pth.tar
--val-cacc