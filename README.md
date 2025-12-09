# Sagemaker-NeRF
This repository is dedicated to testing various AWS services for training and deploying Neural Radiance Field (NeRF) models. The experiments focus on a synthetic dataset of rendered foraminifera images, created using Blender. The dataset includes multi-view images, corresponding camera paths, and a transforms.json file required for NeRF training. You can access the dataset here: [insert link].

## Requirements

To run the training and deployment workflows in this repository, ensure the following prerequisites are met:

AWS SageMaker Unified Studio with GPU-enabled compute resources.

Amazon ECR (Elastic Container Registry) access with appropriate permissions granted by your AWS administrator.

Docker for Windows (or equivalent) installed locally if you plan to build and push custom containers from your machine into ECR.

An AWS IAM configuration that supports pushing/pulling images and creating SageMaker endpoints.
