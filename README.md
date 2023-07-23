To run this file 


    python -m mains.train --exp_name exp4 --batch_size 1 --epochs 10 --optimizer Adam --lr 0.001 --momentum 0.9 --scheduler step --no_cuda True --eval False --num_points 4096 --dropout 0.2 --emb_dims 1024 --k 20 --model_root '' --num_class 4 --data data/preprocessed --metric IOU

this is the command. Read up about label smoothing, integrating resnet50 architecture into the model.