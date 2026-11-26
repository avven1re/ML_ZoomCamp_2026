set -e
ECR_URL=189624869573.dkr.ecr.ap-southeast-2.amazonaws.com
REPO_URL=${ECR_URL}/straight-prediction
LOCAL_IMAGE=mlzoomcamp-2026-serverless
REGION=ap-southeast-2


aws ecr get-login-password \
  --region ${REGION} \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}


REMOTE_IMAGE_TAG="${ECR_URL}/straight-prediction:v1"

docker build --platform linux/amd64 --provenance=false -t ${LOCAL_IMAGE} .
docker tag mlzoomcamp-2026-serverless 189624869573.dkr.ecr.ap-southeast-2.amazonaws.com/straight-prediction:v2
docker push 189624869573.dkr.ecr.ap-southeast-2.amazonaws.com/straight-prediction:v2
echo "Done"