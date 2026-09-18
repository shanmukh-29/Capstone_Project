# Google Colab Runbook

Open a new Google Colab notebook and run these cells from top to bottom.

## 1. Clone the repository

```python
!git clone https://github.com/shanmukh-29/Capstone_Project.git
%cd Capstone_Project
```

## 2. Install dependencies

```python
!pip install -q -r requirements.txt
```

## 3. Run the data pipeline

```python
!python data_pipeline/pipeline.py
```

Verify its outputs:

```python
from pathlib import Path

for path in [
    Path("data_pipeline/books.db"),
    Path("data_pipeline/clean_books.csv"),
    Path("data_pipeline/query_outputs.txt"),
]:
    print(path, path.exists())
```

## 4. Run analytics

```python
!python analytics/analysis.py
```

Verify its outputs:

```python
for path in [
    Path("analytics/titanic.csv"),
    Path("analytics/eda_report.txt"),
    Path("analytics/classification_metrics.csv"),
    Path("analytics/best_pipeline.joblib"),
]:
    print(path, path.exists())

print(*sorted(Path("analytics/figures").glob("*.png")), sep="\n")
```

## 5. Test the support assistant

```python
!python support_assistant/main.py
```

## 6. Start the API in Colab

```python
!uvicorn support_assistant.main:app --host 0.0.0.0 --port 8000 &
```

The API can be tested inside Colab with:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"query": "What is the delivery policy?"},
)
print(response.json())
```

Colab has internet access, so the books scrape and the initial Titanic download should work. Generated artifacts remain in the Colab runtime until downloaded or committed back to GitHub.
