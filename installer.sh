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
DIR=./vicuna-7B
if [ -d "$DIR" ];
then
    echo "$DIR directory exists."
else
    wget https://llm-models-0.s3.us-east-2.amazonaws.com/vicuna-7B.zip
    jar xvf vicuna-7B.zip
    rm -f vicuna-7B.zip
fi
npm install
npm run build
pip3 install --user -r requirements.txt
# flask --app=server.py run --host=0.0.0.0 --port=80