FROM tensorflow/tensorflow:latest-gpu

RUN pip install catboost==0.24.3 xgboost==1.2.1 scikit-learn==0.23.2 lightgbm==2.3.1 flaml
RUN pip install pandas timm

RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch_lightning