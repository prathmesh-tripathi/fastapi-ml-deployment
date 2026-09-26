                   HTTP Request
                        │
                        ▼
                ┌──────────────┐
                │   FastAPI    │
                └──────┬───────┘
                       │
                       ▼
                 Pydantic validation
                       │
                       ▼
                  Feature vector
                       │
                       ▼
              ┌─────────────────┐
              │ Random Forest   │
              │ iris_model.joblib│
              └────────┬────────┘
                       │
                       ▼
                    Prediction
                       │
                       ▼
                  JSON response

- Recommend python -m pip rather than just pip because it guarantees you're using the pip associated with the active Python interpreter.