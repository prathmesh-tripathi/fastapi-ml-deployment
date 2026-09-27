# FastAPI ML Deployment — End-to-End Tutorial

A hands-on tutorial for building, containerizing, version-controlling, and deploying a machine-learning inference API using:

- Python
- scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Docker
- Git/GitHub
- Railway

The project uses the classic **Iris dataset** and a `RandomForestClassifier`.

The goal is not just to build an ML API, but to understand the complete path:

```text
ML Training
    ↓
Model Artifact
    ↓
FastAPI API
    ↓
Docker Container
    ↓
Git/GitHub
    ↓
Railway
    ↓
Public ML API
    ↓
Automatic Redeployment
    ↓
Reproducible ML Environment
```

---

# 1. What You Will Learn

By completing this tutorial, you will learn how to:

1. Train a simple machine-learning model.
2. Save a trained model using Joblib.
3. Build a FastAPI inference API.
4. Validate API input using Pydantic.
5. Run the API locally with Uvicorn.
6. Containerize the application with Docker.
7. Build and run the Docker image locally.
8. Create a proper `.dockerignore`.
9. Create a proper `.gitignore`.
10. Commit the project to Git.
11. Push the project to GitHub.
12. Connect a GitHub repository to Railway.
13. Deploy the Dockerized API to Railway.
14. Expose the API using a public Railway domain.
15. Automatically redeploy the application after a GitHub push.
16. Record the environment used to train the ML model.
17. Understand why serialized ML models can break across library versions.
18. Build a more reproducible ML deployment workflow.

---

# 2. Project Architecture

The project is structured as follows:

```text
fastapi-ml-deployment/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── model.py
│
├── models/
│   ├── iris_model.joblib
│   └── metadata.json
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── train.py
```

## Purpose of Each File

| File | Purpose |
|---|---|
| `train.py` | Trains and saves the ML model |
| `models/iris_model.joblib` | Serialized trained model |
| `models/metadata.json` | Training environment/model metadata |
| `app/model.py` | Loads the trained model |
| `app/main.py` | FastAPI application and endpoints |
| `requirements.txt` | Python dependency versions |
| `Dockerfile` | Instructions for building the production container |
| `.dockerignore` | Files excluded from Docker build context |
| `.gitignore` | Files excluded from Git |
| `README.md` | Project documentation |

---

# 3. Prerequisites

You should have the following installed:

- Python 3.12
- pip
- Git
- Docker Desktop
- A GitHub account
- A Railway account

Verify Python:

```powershell
python --version
```

Verify Git:

```powershell
git --version
```

Verify Docker:

```powershell
docker --version
```

---

# 4. Create the Project

Create a project directory:

```powershell
mkdir fastapi-ml-deployment
cd fastapi-ml-deployment
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Your terminal should now show something similar to:

```text
(.venv) PS C:\Users\...\fastapi-ml-deployment>
```

---

# 5. Install Dependencies

Install the required packages:

```powershell
pip install fastapi uvicorn scikit-learn joblib
```

The main packages are:

### FastAPI

Used to create the REST API.

### Uvicorn

Runs the FastAPI application.

### scikit-learn

Used to train the machine-learning model.

### Joblib

Used to serialize and load the trained model.

---

# 6. Train the Machine-Learning Model

Create:

```text
train.py
```

Add:

```python
from pathlib import Path
import json
import platform

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


iris = load_iris()

X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)

print(f"Model accuracy: {accuracy:.2%}")


models_dir = Path("models")
models_dir.mkdir(exist_ok=True)


joblib.dump(
    model,
    models_dir / "iris_model.joblib",
)

print("Model saved to models/iris_model.joblib")


metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "joblib_version": joblib.__version__,
    "model_type": type(model).__name__,
}


with open(models_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)


print("Metadata saved to models/metadata.json")
```

Run:

```powershell
python train.py
```

Expected output:

```text
Model accuracy: 90.00%
Model saved to models/iris_model.joblib
Metadata saved to models/metadata.json
```

This creates:

```text
models/
├── iris_model.joblib
└── metadata.json
```

---

# 7. Why Model Metadata Matters

Machine-learning models are not always portable between arbitrary versions of Python libraries.

For example, a model trained with:

```text
scikit-learn 0.24.1
```

may fail when loaded with:

```text
scikit-learn 1.6.1
```

A serialized model can contain internal structures whose representation changes between library versions.

A previous example produced an error similar to:

```text
ValueError: node array from the pickle has an incompatible dtype
```

The important lesson is:

> A model artifact is not necessarily independent of the environment in which it was created.

Therefore we record metadata:

```json
{
    "python_version": "3.12.x",
    "scikit_learn_version": "1.x.x",
    "joblib_version": "1.x.x",
    "model_type": "RandomForestClassifier"
}
```

This does not itself make the model compatible.

Instead, it tells us what environment was used to create the model.

---

# 8. Generate `requirements.txt`

After installing the dependencies, generate:

```powershell
python -m pip freeze > requirements.txt
```

View it:

```powershell
type requirements.txt
```

It will contain the exact installed package versions.

For example:

```text
fastapi==...
joblib==...
scikit-learn==...
uvicorn==...
```

The exact versions depend on the environment in which the project was created.

## Why Pin Dependencies?

Without version pinning:

```text
scikit-learn
fastapi
joblib
```

a future installation could retrieve newer versions.

With pinned versions:

```text
scikit-learn==specific.version
joblib==specific.version
```

the deployment environment is much more predictable.

---

# 9. Build the FastAPI Application

Create:

```text
app/
```

Inside it create:

```text
app/__init__.py
```

This can remain empty.

---

# 10. Load the Model

Create:

```text
app/model.py
```

Add:

```python
import joblib


MODEL_PATH = "models/iris_model.joblib"

model = joblib.load(MODEL_PATH)
```

This loads the model when the application starts.

---

# 11. Create the FastAPI Application

Create:

```text
app/main.py
```

Add:

```python
from fastapi import FastAPI
from pydantic import BaseModel

from app.model import model


app = FastAPI(
    title="Iris Prediction API",
    description="A simple ML prediction API using FastAPI",
    version="1.1.0",
)


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {
        "message": "Iris Prediction API is running",
        "version": "1.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }
```

---

# 12. Run the API Locally

Start Uvicorn:

```powershell
python -m uvicorn app.main:app --reload
```

The API should be available at:

```text
http://127.0.0.1:8000
```

Open:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
    "message": "Iris Prediction API is running",
    "version": "1.1.0"
}
```

---

# 13. FastAPI Swagger Documentation

FastAPI automatically generates interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You should see:

```text
GET  /
GET  /health
POST /predict
```

The `/docs` interface allows you to test the API directly from your browser.

---

# 14. Test the Prediction Endpoint

Open:

```text
http://127.0.0.1:8000/docs
```

Select:

```text
POST /predict
```

Click:

```text
Try it out
```

Use:

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

Expected prediction:

```json
{
    "prediction": 0
}
```

The Iris dataset classes are:

```text
0 → setosa
1 → versicolor
2 → virginica
```

---

# 15. Dockerize the Application

Create a file named:

```text
Dockerfile
```

Use:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

# 16. Understand the Dockerfile

## Base Image

```dockerfile
FROM python:3.12-slim
```

Uses Python 3.12 with a relatively small Linux image.

---

## Working Directory

```dockerfile
WORKDIR /app
```

All following commands operate from `/app`.

---

## Copy Dependencies First

```dockerfile
COPY requirements.txt .
```

Then:

```dockerfile
RUN pip install --no-cache-dir -r requirements.txt
```

This is useful because Docker can cache the dependency installation layer.

If only application code changes, Docker may reuse the dependency layer.

---

## Copy Application Code

```dockerfile
COPY app ./app
COPY models ./models
```

This puts the API code and model inside the image.

---

## Expose Port

```dockerfile
EXPOSE 8000
```

Documents the port used by the application.

---

## Start the Application

```dockerfile
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

The application listens on:

```text
0.0.0.0
```

rather than only localhost.

The expression:

```text
${PORT:-8000}
```

means:

- use the `PORT` environment variable if available
- otherwise use port `8000`

This allows the same container to work locally and in platforms that provide a dynamic port.

---

# 17. Create `.dockerignore`

Create:

```text
.dockerignore
```

Add:

```text
.venv/
__pycache__/
*.pyc
.git/
.gitignore
.env
```

This prevents unnecessary files from being sent to Docker during the build.

---

# 18. Build the Docker Image

Build:

```powershell
docker build -t iris-fastapi .
```

Docker will:

```text
Read Dockerfile
      ↓
Create Python base image
      ↓
Install requirements
      ↓
Copy application
      ↓
Copy model
      ↓
Create final image
```

---

# 19. Run the Docker Container

Run:

```powershell
docker run --rm -p 8000:8000 iris-fastapi
```

Open:

```text
http://127.0.0.1:8000/docs
```

Test `/predict` again.

This confirms that:

```text
Python
+
Dependencies
+
FastAPI
+
Model
```

all work inside the container.

---

# 20. Why Docker Is Important for ML Deployment

Without Docker:

```text
Your computer
├── Python version
├── scikit-learn version
├── joblib version
├── FastAPI version
└── OS configuration
```

Another machine may have different versions.

Docker packages the application environment into an image:

```text
Docker Image
├── Linux environment
├── Python
├── Python packages
├── FastAPI
├── scikit-learn
├── Joblib
├── Application code
└── ML model
```

This greatly improves deployment consistency.

---

# 21. Git Configuration

Create:

```text
.gitignore
```

Use:

```text
# Virtual environment
.venv/
venv/

# Python cache
__pycache__/
*.py[cod]

# Environment variables / secrets
.env
.env.*

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

Notice that we do **not** ignore `models/` in this tutorial.

That is intentional.

For this small exercise, the trained model is stored in Git so the complete deployment artifact is available to Railway.

For larger production models, storing model artifacts in Git may not be appropriate. Object storage or a model registry may be preferable.

---

# 22. Initialize Git

Initialize:

```powershell
git init
```

Check:

```powershell
git status
```

Add files:

```powershell
git add .
```

Before committing, inspect what is staged:

```powershell
git diff --cached --name-only
```

You should see project files but **not `.venv`**.

Commit:

```powershell
git commit -m "Initial FastAPI ML application"
```

Rename the branch:

```powershell
git branch -M main
```

---

# 23. Push to GitHub

Create a GitHub repository:

```text
fastapi-ml-deployment
```

Add the remote:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/fastapi-ml-deployment.git
```

Push:

```powershell
git push -u origin main
```

The repository used during this tutorial is:

```text
https://github.com/prathmesh-tripathi/fastapi-ml-deployment
```

---

# 24. Connect GitHub to Railway

Create a new Railway project.

Choose:

```text
Deploy from GitHub Repo
```

Select:

```text
fastapi-ml-deployment
```

Railway detects the root:

```text
Dockerfile
```

and uses it to build the application.

---

# 25. Railway Deployment

The deployment flow becomes:

```text
GitHub Repository
       ↓
Railway
       ↓
Detect Dockerfile
       ↓
Docker Build
       ↓
Install requirements
       ↓
Copy app
       ↓
Copy model
       ↓
Start Uvicorn
       ↓
Running Container
```

Successful logs should contain messages similar to:

```text
Starting Container
Started server process
Waiting for application startup.
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

---

# 26. Generate a Public Railway Domain

After deployment succeeds, enable public networking for the Railway service and generate a domain.

Your URL will look similar to:

```text
https://your-service.up.railway.app
```

The exact domain is assigned by Railway.

---

# 27. Test the Production API

Open:

```text
https://YOUR-RAILWAY-DOMAIN/
```

Expected:

```json
{
    "message": "Iris Prediction API is running",
    "version": "1.1.0"
}
```

Swagger documentation:

```text
https://YOUR-RAILWAY-DOMAIN/docs
```

Test:

```text
POST /predict
```

with:

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

Expected:

```json
{
    "prediction": 0
}
```

---

# 28. Automatic Deployment

Railway is connected to the GitHub branch.

Therefore:

```text
Modify code
     ↓
git add
     ↓
git commit
     ↓
git push
     ↓
Railway detects GitHub change
     ↓
New deployment
     ↓
New Docker image
     ↓
New container
```

We tested this by changing:

```text
API version 1.0.0
```

to:

```text
API version 1.1.0
```

and changing the root endpoint.

After:

```powershell
git add app/main.py
git commit -m "Update API version to 1.1.0"
git push
```

Railway automatically deployed the new version.

---

# 29. Model Reproducibility

This is one of the most important lessons in the project.

A machine-learning deployment is not simply:

```text
model.pkl
```

Instead, think of it as:

```text
Model Artifact
+
Python Version
+
Library Versions
+
Model Code
+
Training Code
+
Configuration
```

For example:

```text
Python 3.12
scikit-learn X.Y.Z
joblib X.Y.Z
RandomForestClassifier
training code
model artifact
```

These components together describe the environment in which the model was created.

---

# 30. Why Pickle/Joblib Can Break

Python ML models are often serialized using:

```python
joblib.dump(model, "model.joblib")
```

and loaded using:

```python
model = joblib.load("model.joblib")
```

However, serialized objects may depend on the implementation of the library that created them.

For example:

```text
Training environment

Python 3.x
scikit-learn 0.24.1
      ↓
model.joblib
```

Later:

```text
Deployment environment

Python 3.x
scikit-learn 1.6.1
      ↓
joblib.load(...)
      ↓
ERROR
```

One possible error is:

```text
ValueError:
node array from the pickle has an incompatible dtype
```

The exact compatibility behavior depends on the model, library versions, and serialized object.

---

# 31. Model Metadata

This project records metadata in:

```text
models/metadata.json
```

Example:

```json
{
    "python_version": "3.12.x",
    "scikit_learn_version": "1.x.x",
    "joblib_version": "1.x.x",
    "model_type": "RandomForestClassifier"
}
```

This provides a record of the environment used to create the model.

---

# 32. `requirements.txt` vs `metadata.json`

These files serve different purposes.

| File | Purpose |
|---|---|
| `requirements.txt` | Defines dependencies needed by the application |
| `metadata.json` | Records important information about the model/training environment |
| `iris_model.joblib` | Contains the serialized model |

Think of it as:

```text
requirements.txt
       ↓
"What should I install?"

metadata.json
       ↓
"What environment created this model?"

iris_model.joblib
       ↓
"What model should I load?"
```

---

# 33. Reproducibility Checklist

For a production ML project, record:

```text
☑ Python version
☑ Dependency versions
☑ Training code
☑ Model artifact
☑ Model type
☑ Dataset/version
☑ Feature definitions
☑ Target definition
☑ Training parameters
☑ Random seeds
☑ Configuration
☑ Environment variables
☑ Container definition
```

The more important the model, the more important this becomes.

---

# 34. Local → Docker → Cloud

This tutorial follows a useful deployment progression.

## Stage 1 — Local

```text
Python
   ↓
FastAPI
   ↓
Localhost
```

Goal:

> Does the application work?

---

## Stage 2 — Docker

```text
Python
   ↓
Docker
   ↓
FastAPI
   ↓
localhost
```

Goal:

> Does the application work inside a reproducible container?

---

## Stage 3 — GitHub

```text
Local
   ↓
Git
   ↓
GitHub
```

Goal:

> Is the source code version-controlled?

---

## Stage 4 — Railway

```text
GitHub
   ↓
Railway
   ↓
Docker
   ↓
Public API
```

Goal:

> Can the application be deployed and accessed remotely?

---

## Stage 5 — Automatic Deployment

```text
Code change
    ↓
Git push
    ↓
Railway
    ↓
Automatic deployment
```

Goal:

> Can changes move from development to production without manually rebuilding and uploading the application?

---

# 35. API Endpoints

The application exposes:

## `GET /`

Returns basic application information.

Example:

```json
{
    "message": "Iris Prediction API is running",
    "version": "1.1.0"
}
```

---

## `GET /health`

Health check endpoint.

Example:

```json
{
    "status": "healthy"
}
```

---

## `POST /predict`

Accepts Iris measurements and returns a model prediction.

Request:

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

Response:

```json
{
    "prediction": 0
}
```

---

# 36. Useful Commands

## Activate virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

## Install dependencies

```powershell
pip install fastapi uvicorn scikit-learn joblib
```

## Generate requirements

```powershell
python -m pip freeze > requirements.txt
```

## Train model

```powershell
python train.py
```

## Run FastAPI locally

```powershell
python -m uvicorn app.main:app --reload
```

## Build Docker image

```powershell
docker build -t iris-fastapi .
```

## Run Docker container

```powershell
docker run --rm -p 8000:8000 iris-fastapi
```

## Git status

```powershell
git status
```

## Stage changes

```powershell
git add .
```

## Commit

```powershell
git commit -m "Description of change"
```

## Push

```powershell
git push
```

---

# 37. Common Problems

## Problem 1 — `.venv` appears in Git

Make sure `.gitignore` contains:

```text
.venv/
venv/
```

If files were already staged:

```powershell
git reset
```

Then:

```powershell
git add .
```

Check:

```powershell
git diff --cached --name-only
```

---

# 38. Problem 2 — Model Loading Error

If you see an error similar to:

```text
InconsistentVersionWarning
```

or:

```text
ValueError:
node array from the pickle has an incompatible dtype
```

check:

```text
Python version
scikit-learn version
joblib version
```

Compare the training environment with the deployment environment.

Do not assume that a serialized model created with one scikit-learn version will always load successfully with another version.

---

# 39. Problem 3 — Docker Cannot Find the Model

Make sure the Dockerfile contains:

```dockerfile
COPY models ./models
```

and that the model actually exists:

```text
models/
└── iris_model.joblib
```

---

# 40. Problem 4 — FastAPI Cannot Be Reached from Docker

Make sure Uvicorn binds to:

```text
0.0.0.0
```

not:

```text
127.0.0.1
```

Correct:

```text
--host 0.0.0.0
```

---

# 41. Problem 5 — Railway Deployment Works but API Is Not Public

A running Railway service does not necessarily mean that you already have a public URL.

Check the Railway service's networking settings and generate a public domain.

---

# 42. Production Considerations

This tutorial intentionally keeps the architecture simple.

A production ML system would typically require additional considerations.

### Model storage

Instead of committing large models to Git:

```text
Git
   ↓
Source code only
```

Model artifacts could be stored in:

```text
Object storage
Model registry
Artifact repository
```

---

### Model versioning

A production system might use:

```text
model-v1
model-v2
model-v3
```

and track:

```text
model version
training dataset
training date
code version
dependency versions
metrics
```

---

### Configuration

Production applications should avoid hardcoding secrets.

Use environment variables for things such as:

```text
API keys
Database URLs
Authentication secrets
External service credentials
```

Never commit secrets to Git.

---

### Health checks

The application already includes:

```text
GET /health
```

A production deployment can use this to determine whether the application is responding correctly.

---

# 43. Final Architecture

After completing this tutorial, the architecture looks like:

```text
                 ┌──────────────────┐
                 │    train.py      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Iris ML Model    │
                 │ Random Forest    │
                 └────────┬─────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
     iris_model.joblib        metadata.json
              │
              └───────────┬───────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    FastAPI       │
                 │    /predict      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │      Docker      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │     GitHub       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │     Railway      │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   Public API     │
                 └──────────────────┘
```

---

# 44. Key Lessons

The most important lessons from this project are:

### 1. An ML model is an artifact

```text
model ≠ just a file
```

The environment that created it matters.

### 2. Pin dependencies

Use:

```powershell
python -m pip freeze > requirements.txt
```

when you want to reproduce the current environment.

### 3. Docker improves consistency

Docker packages the runtime environment with the application.

### 4. Git provides version history

Every meaningful application change should be committed.

### 5. GitHub provides a central source repository

The deployment platform can use the repository as the source of truth.

### 6. Railway can automate deployment

A Git push can trigger a new deployment.

### 7. Model compatibility matters

A model trained with one version of scikit-learn may not be safely loadable with another version.

### 8. Metadata improves traceability

Record the environment and important model information when creating the model.

---

# 45. Next Steps

The next exercises can extend this project into a more realistic ML deployment system.

Possible next steps:

```text
1. Simulate a scikit-learn version mismatch
        ↓
2. Fix it using dependency pinning
        ↓
3. Add model-version validation
        ↓
4. Add prediction labels instead of numeric classes
        ↓
5. Add structured API responses
        ↓
6. Add logging
        ↓
7. Add automated tests
        ↓
8. Add CI/CD with GitHub Actions
        ↓
9. Add environment variables
        ↓
10. Separate development and production configuration
        ↓
11. Add model versioning
        ↓
12. Explore Vercel vs Railway deployment architecture
```

---

# 46. What This Project Demonstrates

This project demonstrates a complete basic ML deployment workflow:

```text
                  MACHINE LEARNING
                        │
                        ▼
                   Train Model
                        │
                        ▼
                 Save Model Artifact
                        │
                        ▼
                   FastAPI API
                        │
                        ▼
                     Docker
                        │
                        ▼
                     GitHub
                        │
                        ▼
                    Railway
                        │
                        ▼
                  Public ML API
                        │
                        ▼
                Automatic Deployment
```

This is the foundation for building more sophisticated ML inference services.
