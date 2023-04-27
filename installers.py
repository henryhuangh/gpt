import os
from sys import platform

if platform == "linux" or platform == "linux2":
    # os.system("source activate pytorch")
    # if not os.path.isdir("gpt"):
    #     os.system("git clone https://github.com/henryhuangh/gpt_chatbot.git")

    # else:
    #     os.system("cd gpt")
    os.system("git fetch origin")
    os.system("git merge origin")
    os.system("curl --silent --location https://rpm.nodesource.com/setup_16.x | bash -")
    os.system("yum -y install nodejs")
    if not os.path.isdir("vicuna-7B"):
        os.system("mkdir vicuna-7B")
        os.system("aws s3 cp s3://llm-models-0/vicuna-7B/ ./vicuna-7B --recursive")
    # npm
    os.system("npm install")
    os.system("npm run build")
elif platform == "win32":
    pass


os.system("pip3 install --user -r requirements.txt")
