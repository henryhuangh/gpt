import os
from sys import platform

if platform == "linux" or platform == "linux2":
    os.system("mkdir vicuna-7B")
    os.system("aws s3 cp s3://llm-models-0/vicuna-7B/ ./vicuna-7B --recursive")
elif platform == "win32":
    pass


os.system("python -m pip install -r requirements.txt")
os.system("python -m pip install --force-reinstall  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# save model
embed_model.save('all-mpnet-base-v2')