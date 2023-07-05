# Neural Graph Collaborative Filtering

## Hướng dẫn cài đặt môi trường và chạy code
* Cài đặt conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
* Tạo môi trường conda:
```bash
conda create --name ngcf python=3.9
```
* Kích hoạt môi trường:
```bash
conda activate ngcf
```
* Cài các thư viện cần thiết: 
```bash
pip install numpy scipy scikit-learn 
```
* Cài đặt Pytorch:
    ```bash
    # CUDA 10.x
    pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102

    # CUDA 11.x
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

## Thí nghiệm training với các thiết lập khác nhau
Tất cả các thí nghiệm được tiến hành trên tập Gowalla dataset.
* **Thí nghiệm 1**: Train với cấu hình mặc định để xác thực kết quả có giống với paper:
```bash
cd NGCF
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 0
```

* **Thí nghiệm 2**: Sử dụng Hàm kích hoạt Gelu thay cho Leaky Relu mặc định.
```bash
cd NGCF
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 0 --prefix gelu
```

* **Thí nghiệm 3**: Thay đổi learning rate từ 0.0001 thành 0.001.
```bash
cd NGCF
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 0 --prefix lr0.001
```

* **Thí nghiệm 4**: Không sử dụng tín hiệu từ embedding gôc của user & item ở bước concat cuối cùng.
```bash
cd NGCF
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 0 --prefix no-skip-connection
```

* **Thí nghiệm 5**: Không sử dụng tín hiệu self-connection của chính user & item bên trong các kết nối bậc cao.
```bash
cd NGCF
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 0 --prefix no-bi
```

## Test lại các kết quả với các bộ tham số đã train.
```bash
# chọn --prefix ứng với weights của thí nghiệm tương ứng. VD: no-bi là sử dụng weights của thí nghiệm 5.
# chọn --user_test_range là user trong danh sách muốn test. VD: [1,2] là muốn test user với ID = 1.
cd NGCF
python main.py --user_test_range [1,2] --mode test --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --gpu_id 0 --prefix no-bi
```

## Các kết quả đã test trên tập Gowalla dataset.
Các kết quả này được tiến hành trên GPU NVIDIA RTX A4000
```bash
# default setting.
Best Iter=[399]@[49427.8]       recall=[0.15483 0.21863 0.26547 0.30263 0.33289], precision=[0.04756    0.03382 0.02755 0.02371 0.02100], hit=[0.53600  0.64254 0.70306 0.74218 0.77276], ndcg=[0.13157 0.15150 0.16541       0.17584 0.18399]
Total time:  49425.70383524895
```

```bash
# gelu
Best Iter=[388]@[49526.1]       recall=[0.15336 0.21498 0.26085 0.29727 0.32861], precision=[0.04690    0.03326 0.02709 0.02331 0.02071], hit=[0.53332  0.63789 0.69697 0.73779 0.76800], ndcg=[0.12964 0.14901 0.16266       0.17290 0.18129]
Total time:  49523.21150350571
```

```bash
# lr=0.001
Best Iter=[55]@[49978.3]        recall=[0.14118 0.20029 0.24402 0.28060 0.31103], precision=[0.04334    0.03104 0.02538 0.02198 0.01958], hit=[0.50928  0.61581 0.67637 0.71914 0.75122], ndcg=[0.11752 0.13618 0.14918       0.15941 0.16756]
Total time:  49975.246522665024
```

```bash
# no origin embedding in final concat
save the latest weights in path:  model/no-skip-connection_gowalla_latest.pkl
Best Iter=[368]@[48633.1]       recall=[0.06553 0.10848 0.14421 0.17540 0.20268], precision=[0.01960    0.01649 0.01481 0.01359 0.01262], hit=[0.28039  0.40003 0.47887 0.53580 0.58182], ndcg=[0.04531 0.05962 0.07043       0.07918 0.08643]
Total time:  48630.57800912857
```

```bash
# no bi embedding in sum
Best Iter=[396]@[48815.2]       recall=[0.15527 0.21857 0.26428 0.30180 0.33144], precision=[0.04755    0.03376 0.02741 0.02364 0.02091], hit=[0.53919  0.64492 0.70413 0.74663 0.77323], ndcg=[0.13223 0.15204 0.16561       0.17617 0.18415]
Total time:  48812.18077349663
```