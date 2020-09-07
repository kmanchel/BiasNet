# BiasNet
Machine Learning as a microservice on Political Bias and Hyperpartisan Detection

## Layout
```commandline
.
├── LICENSE
├── Makefile
├── README.md
├── bin
│   └── create-conda-env.sh
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
│       ├── __init__.py
│       └── __pycache__
├── doc
├── docker
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── docs
├── environment.yml
├── notebooks
│   └── LR_BoW.ipynb
├── requirements.txt
├── results
│   ├── models
│   │   ├── BoW_tokens.pkl
│   │   └── LR_BoW_test.joblib
│   └── pipelines
├── src
│   ├── __init__.py
│   ├── data
│   ├── deployment
│   │   ├── app.py
│   │   ├── request.py
│   │   ├── static
│   │   │   └── css
│   │   │       └── style.css
│   │   └── templates
│   │       └── index.html
│   ├── models
│   │   └── logisticRegression_BoW.py
│   └── preprocess
│       ├── BoW_pipeline.py
│       ├── __pycache__
│       └── pipeline.joblib
└── tools

```

## API Deployment

To deploy the app using Docker, please execute the following commands:
```commandline
cd ./docker
docker-compose up --build
```
