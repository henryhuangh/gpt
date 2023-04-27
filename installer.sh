#!/bin/bash
# run on ec2 g4 instance Deep Learning AMI GPU PyTorch 2.0.0
#
#
# source activate pytorch
#
# git clone https://github.com/henryhuangh/gpt_chatbot.git
# cd gpt_chatbot

git fetch origin
git merge origin
curl --silent --location https://rpm.nodesource.com/setup_16.x | bash -
yum -y install nodejs
DIR=/vicuna-7B
if [ -d "$DIR" ];
then
    echo "$DIR directory exists."
else
    mkdir vicuna-7B
    aws s3 cp s3://llm-models-0/vicuna-7B/ ./vicuna-7B --recursive
fi
npm install
npm run build
pip3 install --user -r requirements.txt
# flask --app=server.py run --host=0.0.0.0 --port=80