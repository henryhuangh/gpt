# Getting Started

deploy on ec2 g4 instance Deep Learning AMI GPU PyTorch 2.0.0.

in the terminal run

`source activate pytorch`

`git clone https://github.com/henryhuangh/gpt_chatbot.git`

`cd gpt_chatbot`

`sh installer.sh`

`flask --app=server.py run --host=0.0.0.0 --port=80`

# Setup EC2 Instance

It will need ports 80, 22 and 443 open. Configure this in EC2 security groups.
