import os
from sys import platform

if platform == "linux" or platform == "linux2"
    os.system("curl --silent --location https://rpm.nodesource.com/setup_16.x | bash -")
    os.system("yum -y install nodejs")
    os.system("mkdir vicuna-7B")
    os.system("aws s3 cp s3://llm-models-0/vicuna-7B/ ./vicuna-7B --recursive")
elif platform == "win32":
    pass

os.system("source activate pytorch")
os.system("pip3 install --user -r requirements.txt")

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# save model
embed_model.save('all-mpnet-base-v2')