# Toxicity Demo Project

Simple demo project for:
- Frontend the user sees
- Backend that receives text/comments
- Model usage through Python

## Project structure

```text
toxicity_demo_project/
├── app.py
├── model.pkl              # put your real trained model here
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## What this does

### Single text mode
Send one comment and get:
- Toxic / Not Toxic

### Batch mode
Send multiple comments and get:
- total comments
- toxic comments
- toxicity percent
- recommendation
- per-comment result

## How to run

1. Open this folder in VS Code
2. Open terminal in the project folder
3. Install packages:

```bash
pip install -r requirements.txt
```

4. Run backend:

```bash
python app.py
```

5. Open browser at:

```text
http://127.0.0.1:5000
```

## Important: your real model

Right now the app has a fallback demo classifier so it works immediately.

When your real model is ready:
- train it in Google Colab
- save it as `model.pkl`
- place `model.pkl` in the project root
- rerun the server

## Expected model format

The backend assumes your model supports:

```python
model.predict(list_of_texts)
```

and returns:
- `0` for Not Toxic
- `1` for Toxic

## Example if you trained with scikit-learn

In Colab:

```python
import joblib
joblib.dump(model, "model.pkl")
```

Then download that file and put it next to `app.py`.
