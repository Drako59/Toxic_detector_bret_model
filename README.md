# Toxicity Detector - BERT Model

A machine-learning project for detecting toxic language in text and YouTube comments.  
The project includes a BERT-based toxicity model notebook and a Flask demo application that exposes the model through a simple web interface.

## Overview

This project analyzes text and returns toxicity scores for multiple toxicity categories. It can be used to test a single comment, analyze a manual list of comments, scan comments from a YouTube video, or scan recent videos from a YouTube channel.

The main goal of the project is to combine natural language processing, model training, and a usable web demo into one practical toxicity-analysis system.

## Features

- Analyze a single text/comment.
- Analyze multiple comments at once.
- Scan comments from a YouTube video.
- Scan recent videos from a YouTube channel.
- Return percentage scores for each toxicity label.
- Detect whether the text is considered toxic based on a threshold.
- Use a trained BERT model when available.
- Use a fallback rule-based model when the trained model cannot be loaded.
- Web interface built with Flask, HTML, CSS, and JavaScript.

## Toxicity Labels

The model works with the following labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

## Tech Stack

- Python
- Flask
- PyTorch
- Hugging Face Transformers
- BERT
- HTML / CSS / JavaScript
- YouTube Data API
- Google Colab / Jupyter Notebook

## Project Structure

```text
Toxic_detector_bret_model/
├── ToxicComment_Bert_Model.ipynb      # Notebook for training/testing the BERT model
├── toxicity_demo_project/
│   ├── app.py                         # Flask backend
│   ├── requirements.txt               # Basic Python requirements
│   ├── templates/
│   │   └── index.html                 # Frontend page
│   └── static/
│       ├── style.css                  # Frontend styling
│       └── script.js                  # Frontend logic
└── README.md
```

## How It Works

1. The user enters text, comments, a YouTube video URL, or a YouTube channel URL.
2. The Flask backend receives the request.
3. The text is tokenized with a BERT tokenizer.
4. The model returns raw logits.
5. The logits are converted into percentages using sigmoid.
6. The app returns scores for all six toxicity labels.
7. If at least one label passes the threshold, the comment is marked as toxic.

## Installation

Clone the repository:

```bash
git clone https://github.com/Drako59/Toxic_detector_bret_model.git
cd Toxic_detector_bret_model/toxicity_demo_project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install flask requests torch transformers
```

You can also install the existing requirements file first:

```bash
pip install -r requirements.txt
```

> Note: the current `requirements.txt` may not include every package used by `app.py`, so installing `requests`, `torch`, and `transformers` may still be required.

## Running the Demo

From inside the `toxicity_demo_project` folder, run:

```bash
python app.py
```

Then open the app in your browser:

```text
http://127.0.0.1:5000
```

## Using the Trained BERT Model

The Flask app expects a saved Hugging Face model folder.  
After training/exporting the model, place the saved model folder locally and update `MODEL_PATH` inside `app.py`.

A typical saved model folder should contain files such as:

```text
toxic_bert_model/
├── config.json
├── model.safetensors or pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

Example export from Python/Colab:

```python
model.save_pretrained("toxic_bert_model")
tokenizer.save_pretrained("toxic_bert_model")
```

## YouTube Analysis

The app supports three YouTube-related workflows:

- Extract and analyze comments from a single YouTube video.
- Analyze a manual list of comments.
- Resolve a YouTube channel and scan comments from its recent videos.

For YouTube scanning, a YouTube Data API key is required.

## API Endpoints

### `POST /predict`

Analyze a single text.

```json
{
  "text": "Example comment"
}
```

### `POST /analyze-comments`

Analyze a list of comments.

```json
{
  "comments": [
    "First comment",
    "Second comment"
  ]
}
```

### `POST /analyze-youtube`

Analyze comments from a YouTube video.

```json
{
  "videoUrl": "https://www.youtube.com/watch?v=VIDEO_ID",
  "maxComments": 100
}
```

### `POST /analyze-youtube-channel`

Analyze recent videos from a YouTube channel.

```json
{
  "channelUrl": "https://www.youtube.com/@channelname",
  "maxVideos": 5,
  "maxCommentsPerVideo": 100
}
```

## Security Note

Do not commit real API keys, tokens, or secrets to GitHub.  
For a production version, store the YouTube API key in an environment variable, for example:

```python
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
```

## Possible Future Improvements

- Add a complete `requirements.txt` for the BERT demo.
- Move configuration values to environment variables.
- Add model accuracy/evaluation results to the README.
- Add screenshots or GIFs of the web interface.
- Add Docker support.
- Improve error handling for YouTube API limits.
- Add unit tests for URL parsing and toxicity scoring.

## Disclaimer

This project is for learning and demonstration purposes. Toxicity detection models can make mistakes, especially with sarcasm, slang, context, and different languages. The results should not be used as the only source for moderation decisions.
