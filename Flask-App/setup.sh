#!/bin/bash -l

export PROJECT_ID='cedar-talent-279001'
gcloud builds --project $PROJECT_ID  submit --tag gcr.io/cedar-talent-279001/feed-rank-service .
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

