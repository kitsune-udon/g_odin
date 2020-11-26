FROM idein/pytorch:latest
RUN pip install -U pip
RUN pip install pytorch-lightning==0.9.0 pytorch-lightning-bolts==0.2.2 tensorboard==2.2.0 scikit-learn==0.23.2
RUN pip install kornia@git+https://github.com/kornia/kornia@e18e682858c57124d46c40e8ab1d136c009a1f69
RUN pip install einops
RUN apt-get update -y
RUN apt-get install patch -y
RUN pip install lmdb
