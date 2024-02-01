# Master Thesis 2024 by August Semrau
This project works with LLMs. More info will be update as progress is made.  
Deadline for thesis is 22st of June 2024.

## Setup
To setup the project, run the following commands:

```bash
# Clone the repository
git clone https://github.com/augustsemrau/thesis2024.git

# Create virtual environment
make create_environment

# Activate virtual environment
conda activate src

# Install dependencies
make requirements

# Install dev dependencies
make requirements_dev

# pip install the project
pip install -e .
```

## OpenAI API Key
To use the OpenAI API, you need to set the environment variable OPENAI_API_KEY, like so:
```bash
export OPENAI_API_KEY=<your key here>
```


## Data
In order to create the relevant dataset, you need to:
1. Download the raw data and place it in the data/raw folder. A link for the raw data will be provided later.
2. Run the following command from the root folder (thesis2024):
```bash
python src/thesis2024/datamodules/make_dataset.py
```
This will convert all raw data into a vector store using Chroma.


## Running the code
The first chatbot prototype can be run using the following command from the root folder (thesis2024):
```bash
streamlit run src/thesis2024/models/chatbot_v1.py
```


## Project structure
The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```