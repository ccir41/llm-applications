## Create virtual environment
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

## Obtain Pinecone API Key
- Signup for pinecone and obtaine API KEY
- Export it to PINECONE_API_KEY environment variable using below command
    - `export PINECONE_API_KEY=your-pinecone-api-key`

## Indexing csv file description
- `python pinecone_upsert.py`

## Running application
- `streamlit run app.py`

## Running notebook
- `jupyter notebook --port 8888`

## Sample questions for testing
1. What was the labor force participation rate in Maryland in January 2007, and how many people were employed?
2. What are the trends in shooting incidents over the years?
3. What are the trends in different types of crossings (trucks, buses, pedestrians, etc.) over the past five years?
4. Is there a correlation between border crossing entries and the employment rate in border states?