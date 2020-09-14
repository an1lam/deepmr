if [[ $1 == "-f" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse $2 gregor-3-vm:~/project/$3
fi
if [[ $1 == "-n" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/*.ipynb gregor-3-vm:~/project/src/
fi
if [[ $1 == "-p" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/*.py gregor-3-vm:~/project/src/
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/data/*.py gregor-3-vm:~/project/src/data/
fi
if [[ $1 == "-c" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/pyx/*.py gregor-3-vm:~/project/src/pyx/
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/pyx/*.pyx gregor-3-vm:~/project/src/pyx/
fi
if [[ $1 == "-d" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./dat/ gregor-3-vm:~/project/
fi
if [[ $1 == "-s" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./scr/ gregor-3-vm:~/project/
fi
if [[ $1 == "-b" ]]; then
    gcloud compute scp --project coms-4995-260215 --zone us-west1-b --recurse ./src/bpnet/ gregor-3-vm:~/project/
fi
