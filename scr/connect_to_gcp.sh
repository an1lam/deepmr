ZONE=us-west1-b
PROJECT_ID=coms-4995-260215
INSTANCE_NAME=gregor-3-vm
gcloud compute ssh --project $PROJECT_ID --zone $ZONE   $INSTANCE_NAME
