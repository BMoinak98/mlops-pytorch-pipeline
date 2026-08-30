# MLOps PyTorch Kubernetes Pipeline

An end-to-end MLOps pipeline for training, containerizing, and serving a PyTorch computer vision model on Kubernetes using the CIFAR-10 dataset. The repository integrates automated unit testing via GitHub Actions CI, configuration management via Kubernetes ConfigMaps, persistent storage via PVCs, and a production-ready FastAPI RESTful inference service equipped with Horizontal Pod Autoscaling (HPA).

- **Author**: Moinak Bandyopadhyay
- **Repository**: [BMoinak98/mlops-pytorch-pipeline](https://github.com/BMoinak98/mlops-pytorch-pipeline)

---

## 🏗️ System Architecture

```text
                                +-----------------------------+
                                |      Persistent Volume      |
                                |         (mlops-pvc)         |
                                +--------------+--------------+
                                               |
                     +-------------------------+-------------------------+
                     | Mount: /app/data        | Mount: /app/checkpoints |
                     v                         v
        +-------------------------+   Saves   +-------------------------+
        |  pytorch-training-job   |---------->|    classifier_v1.pt     |
        |  (Kubernetes Batch Job) |           +------------+------------+
        +-------------------------+                        |
                                                           | Mount (readOnly)
                                                           v
+-------------------------+                   +-------------------------+
|     model-serving-hpa   |====== Scales =====>      model-serving      |
|    (HPA: 2-5 Replicas)  |                   |  (FastAPI Deployment)   |
+-------------------------+                   +------------+------------+
                                                           ^
                                                           | Routes Traffic
                                              +------------+------------+
                                              |      model-serving      |
                                              |   (ClusterIP Service)   |
                                              +------------+------------+
                                                           |
                                                           | POST /predict
                                                           v
                                                  [ Client / cURL ]
```

---

## 📁 Repository Structure

```plaintext
mlops-pytorch-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI workflow (tests & validation)
├── configs/
│   └── training_config.yaml    # Local / baseline training hyperparameters
├── docker/
│   ├── Dockerfile.serve        # Hardened non-root FastAPI serving image
│   └── Dockerfile.train        # Multi-stage PyTorch model training image
├── k8s/
│   ├── configmap.yaml          # Training configuration mapped into Pods
│   ├── hpa.yaml                # Horizontal Pod Autoscaler for inference service
│   ├── namespace.yaml          # Kubernetes namespace definition (ml-training)
│   ├── serving-deployment.yaml # FastAPI deployment with health probes & PVC mount
│   ├── serving-service.yaml    # ClusterIP Service exposing port 80 -> 8080
│   └── training-job.yaml       # Batch Job definition executing model training
├── requirements/
│   ├── serve.txt               # Dependencies for FastAPI model serving
│   └── train.txt               # Dependencies for PyTorch model training
├── src/
│   ├── __init__.py
│   ├── dataset.py              # CIFAR-10 data loaders & augmentations
│   ├── model.py                # PyTorch ResNet-18 model architecture
│   ├── serve.py                # FastAPI inference server & health endpoints
│   └── train.py                # Training loop, evaluation & early stopping logic
├── tests/
│   └── test_model.py           # Unit tests for architecture & tensor shapes
├── .dockerignore               # Build context exclusions
├── .gitignore                  # Git exclusions
├── pytest.ini                  # Pytest configuration
├── README.md                   # Project documentation
└── test_image.png              # Sample test image for inference verification
```

---

## ⚙️ Configuration Management

Training hyperparameters are decoupled from source code and managed via YAML files:

```yaml
# configs/training_config.yaml
model:
  architecture: resnet18
  num_classes: 10
training:
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
  early_stopping_patience: 3
data:
  dataset: cifar10
  data_dir: /app/data
output:
  checkpoint_dir: /app/checkpoints
  model_name: classifier_v1.pt
```

In Kubernetes environments, this configuration is mounted dynamically via `k8s/configmap.yaml` into `/app/configs/training_config.yaml`.

---

## 🚀 Local Development & Quickstart

### 1. Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install training and serving dependencies
pip install -r requirements/train.txt -r requirements/serve.txt pytest
```

### 2. Run Tests

```bash
pytest tests/
```

### 3. Run Training Locally

```bash
python src/train.py
```

### 4. Run Inference Server Locally

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8080
```

---

## 🐳 Docker Containerization

The project uses modular Docker images tailored for training and serving:

### Build Training Image
```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

### Build Serving Image
```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

---

## ☸️ Kubernetes Deployment

Deploy the end-to-end pipeline onto a Kubernetes cluster (e.g., Minikube, Kind, EKS, GKE):

### 1. Create Namespace & ConfigMap
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
```

### 2. Ensure Persistent Storage (PVC) Exists
Ensure a PersistentVolumeClaim named `mlops-pvc` is bound in the `ml-training` namespace for storing dataset caches and trained checkpoints.

### 3. Launch the Training Job
```bash
kubectl apply -f k8s/training-job.yaml

# Stream training logs
kubectl logs -f -l app=pytorch-training -n ml-training
```

### 4. Deploy the Inference Service & HPA
Once training completes and saves `classifier_v1.pt` into the shared volume:

```bash
# Deploy model serving deployment & ClusterIP service
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml

# Apply Horizontal Pod Autoscaler
kubectl apply -f k8s/hpa.yaml
```

---

## 📡 API Reference & Inference Verification

### Health Check
```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Run Image Classification
Send an image (`test_image.png`) as `multipart/form-data`:

```bash
curl -X POST "http://localhost:8080/predict" \
  -F "image=@test_image.png"
```

**Response:**
```json
{
  "probabilities": {
    "0": 0.0215,
    "1": 0.8124,
    "2": 0.0051,
    "3": 0.0342,
    "4": 0.0118,
    "5": 0.0456,
    "6": 0.0089,
    "7": 0.0152,
    "8": 0.0321,
    "9": 0.0132
  }
}
```

### 🏷️ CIFAR-10 Class Labels Mapping

The predicted output indices (`0` to `9`) correspond to standard CIFAR-10 categories:

| Class Index | Class Name | Description |
| :---: | :--- | :--- |
| `0` | **Airplane** | Aeroplanes, jets, commercial aircraft |
| `1` | **Automobile** | Cars, sedans, SUVs |
| `2` | **Bird** | Birds of all species |
| `3` | **Cat** | Feline domestic pets |
| `4` | **Deer** | Deer, stags, fawns |
| `5` | **Dog** | Canine domestic pets |
| `6` | **Frog** | Amphibians, frogs, toads |
| `7` | **Horse** | Horses, equines |
| `8` | **Ship** | Boats, cargo ships, vessels |
| `9` | **Truck** | Lorries, semi-trailers, pickup trucks |

---

## 🔄 CI/CD Automation

Continuous Integration is automated via GitHub Actions ([.github/workflows/ci.yml](file:///.github/workflows/ci.yml)):
- **Triggers**: Pushes to `main` and `develop`, plus Pull Requests targeting `develop`.
- **Pipeline Tasks**: Checks out code on `ubuntu-latest`, configures Python 3.11, installs all requirements, and executes `pytest tests/`.
