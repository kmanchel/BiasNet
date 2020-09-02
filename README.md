# BiasNet
Machine Learning as a microservice on Political Bias and Hyperpartisan Detection

## Layout
```commandline
.
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
│       └── __init__.py
├── docker
│   ├── DockerFile
│   ├── build_env.sh
│   └── environment.yml
├── docs
├── notebooks
│   └── LR_BoW.ipynb
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
│       └── pipeline.joblib
└── tools

```