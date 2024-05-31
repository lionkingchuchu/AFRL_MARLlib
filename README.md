## Details_1.12
```
conda create -n marllib python=3.8
conda activate marllib
git clone https://github.com/MaDoKaLiF/AFRL_MARLlib.git
pip install --upgrade pip
pip install -r requirements.txt

pip install gym>=0.22.0
cd MARLLIB
python marllib/patch/add_patch.py -y
```
### update 1.23
pip install pettingzoo==1.23.1
pip install supersuit==3.9.0
pip install pygame==2.3.0

### Expected Errors
- gym=0.20.0 설치 과정에서 에러 발생  
  setuptools=65.5.0으로 downgrade
  ```
  pip install setuptools==65.5.0 "wheel<0.40.0"
  ```
  
- (vessl 사용 시) import torch 과정에서 AttributeError 발생  
  global environment의 torch를 삭제하고 진행
### Marllib 버전비교
'''
marllib: for_pettingzoo_1.23.0
marllib_1.12: for_pettingzoo_1.12(original)
marllib_bean: for_pettingzoo_bean(코드 업데이트 진행중)
'''

### MARLlib documentation
https://marllib.readthedocs.io/en/latest/index.html
 

@article{hu2022marllib,
  author  = {Siyi Hu and Yifan Zhong and Minquan Gao and Weixun Wang and Hao Dong and Xiaodan Liang and Zhihui Li and Xiaojun Chang and Yaodong Yang},
  title   = {MARLlib: A Scalable and Efficient Multi-agent Reinforcement Learning Library},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
}

### AFRL MARLLIB documentation

Use quay.io/vessl-ai/ngc-pytorch-kernel:21.04-py3-202301190245 as docker image
```
git clone https://github.com/lionkingchuchu/AFRL_MARLlib.git
cd AFRL_MARLLIB
pip install -r requirements.txt
cd MARLLIB
python main.py or cartpoletest.py
```