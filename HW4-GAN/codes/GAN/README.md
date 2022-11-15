1. Changes in `GAN.py`
   1. 为了能使用命令行参数 `--use_mlp`切换到使用MLP-based-GAN进行训练，为部分函数和类增加了 `use_mlp`这一参数；
   2. `weights_init`中增加了初始化线性层的部分；
2. Changes in `main.py`
   1. 增加了命令行参数 `--use_mlp`，`--interpolation`，`--mode_collapse`以及处理相应情况的代码，可以快速得到作业中相应部分的实验结果。
3. 作业相关部分命令
   1. Interpolation：`python main.py --interpolation --latent_dim 100 --generator_hidden_dim 100`
   2. Mode_collapse: `python main.py --mode_collapse --latent_dim 100 --generator_hidden_dim 100`
   3. MLP-based-GAN: `python main.py --do_train --use_mlp --latent_dim 100`
