# Questions

1. Make necessary modifications to the module structure and fix any bugs such that you can run train.py script from `mains` folder.
```shell
python -m mains.train --data data/preprocessed --epochs 2 --scheduler step
```

2. Try to figure out what arguments to pass to the script in order to start training (please note that all files that you need to run training are in the archive). Run training for some number of epochs.

3. Be prepared to discuss what is model architecture and what is being trained.

4. Think about (and if you have time implement) how you would test the training at the end of epochs.

5. Finally, what improvements would you make to the model and the training procedue?


python -m mains.train --exp_name exp4 --batch_size 1 --epochs 10 --optimizer Adam --lr 0.001 --momentum 0.9 --scheduler step --no_cuda True --eval False --num_points 4096 --dropout 0.2 --emb_dims 1024 --k 20 --model_root '' --num_class 4 --data data/preprocessed --metric IOU