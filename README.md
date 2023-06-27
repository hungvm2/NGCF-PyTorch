# Neural Graph Collaborative Filtering
This is my PyTorch implementation for the paper:

>Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). Neural Graph Collaborative Filtering, [Paper in ACM DL](https://dl.acm.org/citation.cfm?doid=3331184.3331267) or [Paper in arXiv](https://arxiv.org/abs/1905.08108). In SIGIR'19, Paris, France, July 21-25, 2019.

The TensorFlow implementation can be found [here](<https://github.com/xiangwang1223/neural_graph_collaborative_filtering>).

## Introduction
My implementation mainly refers to the original TensorFlow implementation. It has the evaluation metrics as the original project. Here is the example of Gowalla dataset:

```
Best Iter=[38]@[32904.5]	recall=[0.15571	0.21793	0.26385	0.30103	0.33170], precision=[0.04763	0.03370	0.02744	0.02359	0.02088], hit=[0.53996	0.64559	0.70464	0.74546	0.77406], ndcg=[0.22752	0.26555	0.29044	0.30926	0.32406]
```

Hope it can help you!

## Environment Requirement
The code has been tested under Python 3.9. The required packages are as follows:
* pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
* pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
* numpy == 1.24.3
* scipy == 1.10.1
* sklearn == 1.2.2

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in NGCF/utility/parser.py).
* Gowalla dataset
```
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 0

# No weights saving
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --save_flag 0 --pretrain 0 --batch_size 10 --epoch 1 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 0
```

* Amazon-book dataset
```
python main.py --dataset amazon-book --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 200 --verbose 50 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 0
```
## Supplement

* The parameter `negative_slope` of LeakyReLu was set to 0.2, since the default value of PyTorch and TensorFlow is different.
* If the arguement `node_dropout_flag` is set to 1, it will lead to higher calculational cost.

# Our training result:
Device: NVIDIA RTX A4000

## Gowalla dataset
```
lr=0.0001
time per epoch: 123s
Total time:  18643.07929468155
Best Iter=[146]@[18643.1]       recall=[0.14389 0.20078 0.24322 0.27854 0.30800], precision=[0.04429    0.03124 0.02538 0.02187 0.01946], hit=[0.52244  0.62409 0.68347 0.72597 0.75742], ndcg=[0.12395 0.14171 0.15433       0.16419 0.17210]
```

```
lr=0.001
time per epoch: 120s
Best Iter=[36]@[5111.3] recall=[0.13922 0.19748 0.24104 0.27650 0.30624], precision=[0.04268    0.03053 0.02501 0.02162 0.01924], hit=[0.50606  0.61163 0.67222 0.71602 0.74700], ndcg=[0.11774 0.13600 0.14896 0.15889 0.16684]
```