PEM_FILE_PATH=${1:-"~/StephenDAK_Dell.pem"}
EC2_DNS=${2:-"ec2-54-157-160-62.compute-1.amazonaws.com	"}
ssh -i $PEM_FILE_PATH ubuntu@$EC2_DNS
