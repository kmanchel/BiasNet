# BiasNet
**Machine Learning as a microservice on Political Bias and Hyperpartisan Detection**

<span style="color:red;">Please note the following repo is undergoing continuous updates. Please refer to the [Feature Backlog](#Feature-Backlog) for upcoming updates.<\span>

## Deployment (Using Docker)

To deploy the app using Docker, please execute the following commands:
```commandline
cd ./docker
docker-compose up --build
```

## Running locally

To reproduce/run the code, you would have to do the following:

1. Clone Repo in your chosen directory
```commandline
cd <directory_path>
git init
git remote add origin https://github.com/msc11/ie590_project.git
git pull origin master
```

2. Build and activate conda environment
```commandline
sh bin/create-conda-env.sh
```

3. Run Flask App
```commandline
cd src
python deployment/app.py
```

4. App Usage

The App may be used through the API (https:0.0.0.0/predict_API) or through the Web app (https:0.0.0.0/predict_UI)

The API Body (**TO BE UPDATED**) is as follows:

```commandline
REQUEST:
{'model_id': '1',
 'text': ' One man called in to comment, starting off with a criticism of CNN, '
         'lambasting the outlet for a chyron from last week that read, Fiery '
         'but mostly peaceful protests after police shooting, while buildings '
         'burned in the background behind a correspondent reporting from '
         'Kenosha, Wisconsin. The caller went on to refer to Stelter as '
         "'Humpty Dumpty' and 'a stooge,' adding, We all know you're not "
         'reliable. According to Fox News, Stelter admitted the chyron was a '
         "mistake. I don't know who wrote it, probably a young producer who's "
         'trying their best under deadline in a breaking news situation, he '
         'said. That kind of thing becomes easily criticized and probably not '
         'the right banner to put on the screen. Another man called in to '
         'confront Stelter, saying, You guys always talk about how many times '
         'Trump has lied. Ive calculated that, I think with your chyrons '
         '...Yeah, I dont know if theres any journalists left at CNN, but I '
         'know that if I were to estimate, about 300 different distortions or '
         'misinformation that we get out of CNN — and you have to watch them '
         'in the airport, which is harsh — but if you added all that up to 46 '
         'months, it comes out to be 300,000-plus distortions of truth. So, my '
         'thing is, you guys — this is how low you will go, the caller '
         'continued. You went out, and you made lies, and you defamed a child, '
         'referring to Covington (Kentucky) Catholic High School student Nick '
         'Sandmann, with whom the network settled a multimillion-dollar '
         'lawsuit. The caller added, And I dont believe in dividing our '
         'nation, it hurts our great nation, and so CNN is really the enemy of '
         'the truth. Thats my opinion, thank you. '}
RESPONSE:
{'prediction': 'leans Right'}
```
The API expects a model_id which specifies what model to use, and the text to be processed and predicted on.

Models currently available are (**TO BE UPDATED**): (i) [Bag Of Words](https://github.com/kmanchel/BiasNet/blob/master/src/models/BoW) (ii) [Hierarchical Attention Networks for Document Classification](https://github.com/kmanchel/BiasNet/blob/master/src/models/hierarchical_attention)


## Layout
```commandline
.
├── bin
│   ├── create-conda-env.sh
│   └── download_glove.sh
├── data
│   ├── processed
│   │   ├── jan_20_clean.csv
│   │   └── sep_oct_lemmatized.csv
│   └── raw
│       ├── jan_20.csv
│       └── sep_oct_19.csv
├── docker
│   ├── docker-compose.yml
│   ├── DockerFile
│   └── entrypoint.sh
├── environment.yml
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── results
│   ├── BoW
│   │   ├── BoW_tokens.pkl
│   │   └── LR_BoW_test.joblib
│   └── hierarchical_attention
│       ├── checkpoint
│       ├── embedding_matrix.npy
│       ├── history.pkl
│       ├── model_checkpoints
│       │   ├── weights.data-00000-of-00002
│       │   ├── weights.data-00001-of-00002
│       │   └── weights.index
│       ├── params.json
│       └── tokenizer.pkl
├── src
│   ├── deployment
│   │   ├── app.py
│   │   ├── request.py
│   │   ├── static
│   │   │   └── css
│   │   │       └── style.css
│   │   └── templates
│   │       └── index.html
│   ├── models
│   │   ├── BoW
│   │   │   └── logisticRegression_BoW.py
│   │   └── hierarchical_attention
│   │       ├── attention.py
│   │       ├── han.py
│   │       ├── __pycache__
│   │       └── train_han.py
│   ├── preprocess
│   │   ├── BoW
│   │   │   ├── BoW_pipeline.py
│   │   │   ├── pipeline.joblib
│   │   │   └── __pycache__
│   │   ├── hierarchical_attention
│   │   │   ├── build_embedding_matrix.py
│   │   │   ├── build_tokenizer.py
│   │   │   ├── clean_dataset.py
│   │   │   ├── han_pipeline.py
│   │   │   └── __pycache__
│   │   ├── __pycache__
│   │   └── utils.py
│   └── __pycache__
└── tools
    └── csv_build.py

```

## Feature Backlog

1. Monitoring System (training, deployment, inference)
2. BERT Base Model
3. Structured Prediction: Targeted Sentiment Analysis
4. Training Data (.sqlite3) DB access (AWS) 

## References

- https://library.cscc.edu/mediabias/detectors 
- https://www.aclweb.org/anthology/S19-2145.pdf 
- https://pdfs.semanticscholar.org/7238/baa75e78838398a03aa705742004ab068d35.pdf?_ga=2.84089053.57717742.1596399737-319321172.1594922709
- https://pdfs.semanticscholar.org/dfe0/4e7e8fe9776e1623a5e9ce99ca4757181e5f.pdf?_ga=2.155824383.57717742.1596399737-319321172.1594922709
- https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
- https://www.aclweb.org/anthology/P14-1105.pdf
