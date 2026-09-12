hiver-support-agent/
│
├── README.md
├── requirements.txt
├── .env.example
├── run_pipeline.py
├── api.py
│
├── data/
│   ├── raw/
│   │   └── customer_support_on_twitter.csv
│   └── golden_set.csv
│
├── outputs/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── intents.py
│   ├── agent.py
│   ├── evaluator.py
│   └── utils.py
│
└── tests/
    └── test_agent.py
