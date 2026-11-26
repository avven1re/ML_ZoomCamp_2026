
ECR_URL=189624869573.dkr.ecr.ap-southeast-2.amazonaws.com
REPO_URL=${ECR_URL}/churn-prediction-lambda
LOCAL_IMAGE=churn-prediction-lambda
REGION=ap-southeast-2


aws ecr get-login-password \
  --region ${REGION} \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}


REMOTE_IMAGE_TAG="${ECR_URL}/churn-prediction-lambda:v1"

docker build --platform linux/arm64 --provenance=false -t ${LOCAL_IMAGE} .
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

echo "Done"