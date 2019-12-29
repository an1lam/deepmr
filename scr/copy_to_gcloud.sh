gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/*.py gregor-2-vm:~/project/src/
gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/*.ipynb gregor-2-vm:~/project/src/
gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/data/*.py gregor-2-vm:~/project/src/data/
gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/pyx/*.py gregor-2-vm:~/project/src/pyx/
gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/pyx/*.pyx gregor-2-vm:~/project/src/pyx/
if [[ $1 == "-d" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./dat/ gregor-2-vm:~/project/
fi
