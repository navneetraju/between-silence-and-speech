# Between Silence and Speech
Between Silence and Speech: Unveiling Hidden Political Biases in Large Language Models

# Dataset

## Data Generation
The dataset was compiled by leveraging the capabilities of GPT o3-mini-high along with deep research methods. The process involved gathering information from reputable, consensus-based sources such as UNHCR, UCDP, ACLED, the Correlates of War, Amnesty International, and declassified government documents. These sources provided neutral, fact-based data on global conflicts and human rights issues spanning the past 50–100 years. The assistant synthesized key factual points into minimally altered representative statements, which were then organized into a structured CSV format. Additional columns were created to list the countries involved—ensuring consistent naming conventions—and two primary languages corresponding to opposing viewpoints were assigned based on each country’s most spoken or national language.

## Use Data
The CSV in the `data` folder contains the dataset used in the paper. The columns are as follows:

- `statement`: A minimally synthesized representative statement on a controversial geopolitical or historical event.
- `countries`: A list of the countries involved, with each country listed separately and consistently.
- `language1`: The primary or most spoken language of the first country listed, representing one opposing viewpoint.
- `language2`: The primary or most spoken language of the second country listed, representing the opposing viewpoint.

# Environment Setup

## Setup virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
 ```
 
*Note: Alternatively, if you are using PyCharm, you can create a virtual environment by going to `File -> Settings -> Project -> Python Interpreter -> Add New Environment` and selecting the Python version you want to use.*

## Install dependencies
```bash
pip install -r requirements.txt
```

# Research Documentation
The research documentation can be found in the `docs` folder.
