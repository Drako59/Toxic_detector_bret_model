from flask import Flask, render_template, request, jsonify
import os
import re
from urllib.parse import urlparse, parse_qs
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

MODEL_PATH = r"C:\Users\maxka\Downloads\toxicity_demo_project\toxicity_demo_project\toxic_bert_model"
YOUTUBE_API_KEY = "AIzaSyCkKXroQMhWWAdR_b7aFIIwUhdZEJnSvGg"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


class ToxicModel:
    def __init__(self, model_path=MODEL_PATH):
        self.labels = LABELS
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def score_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].tolist()

        scores = {
            label: round(float(prob) * 100, 2)
            for label, prob in zip(self.labels, probs)
        }
        return scores

    def predict_scores(self, texts):
        return [self.score_text(text) for text in texts]


class FallbackToxicModel:
    def __init__(self):
        self.labels = LABELS
        self.rules = {
            "toxic": {"idiot", "stupid", "hate", "trash", "moron", "loser", "dumb", "garbage"},
            "severe_toxic": {"kill yourself", "die", "worthless", "go die"},
            "obscene": {"damn", "shit", "fuck", "bitch", "asshole"},
            "threat": {"kill", "hurt", "destroy", "attack", "beat you"},
            "insult": {"ugly", "clown", "pathetic", "loser", "idiot"},
            "identity_hate": {"racist", "nazi", "terrorist", "slur_placeholder"},
        }

    def score_text(self, text):
        lowered = text.lower()
        scores = {}
        for label in self.labels:
            matches = sum(1 for token in self.rules[label] if token in lowered)
            if matches == 0:
                score = 3.0 if len(lowered) > 10 else 0.5
            elif matches == 1:
                score = 72.0
            else:
                score = 93.0
            scores[label] = round(score, 2)
        return scores

    def predict_scores(self, texts):
        return [self.score_text(text) for text in texts]


def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return ToxicModel(MODEL_PATH)
        except Exception as e:
            print(f"Could not load transformer model from {MODEL_PATH}. Falling back. Error: {e}")
    else:
        print(f"{MODEL_PATH} not found. Using fallback model.")
    return FallbackToxicModel()


model = load_model()


def summarize_scores(scores, threshold=50.0):
    active_labels = [label for label, value in scores.items() if value >= threshold]
    overall_toxic = any(value >= threshold for value in scores.values())
    return {
        "overallToxic": overall_toxic,
        "activeLabels": active_labels,
        "scores": scores
    }


def aggregate_label_percentages(per_comment_scores, threshold=50.0):
    total = len(per_comment_scores)
    if total == 0:
        return {label: 0.0 for label in LABELS}

    summary = {}
    for label in LABELS:
        count = sum(1 for scores in per_comment_scores if scores[label] >= threshold)
        summary[label] = round((count / total) * 100, 2)
    return summary


def extract_video_id(url_or_id):
    value = (url_or_id or "").strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    try:
        parsed = urlparse(value)
    except Exception:
        return None

    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if "youtube.com" in host:
        if path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if path.startswith("/shorts/"):
            parts = path.split("/")
            return parts[2] if len(parts) > 2 else None
        if path.startswith("/embed/"):
            parts = path.split("/")
            return parts[2] if len(parts) > 2 else None

    if "youtu.be" in host:
        parts = path.strip("/").split("/")
        return parts[0] if parts and parts[0] else None

    return None


def get_youtube_comments(video_url, max_comments=100):
    if not YOUTUBE_API_KEY:
        raise ValueError("Missing YOUTUBE_API_KEY environment variable.")

    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Could not extract a valid YouTube video ID from the URL.")

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        batch_size = min(100, max_comments - len(comments))
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": batch_size,
            "textFormat": "plainText",
            "key": YOUTUBE_API_KEY
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(
            "https://www.googleapis.com/youtube/v3/commentThreads",
            params=params,
            timeout=25
        )

        try:
            data = response.json()
        except Exception:
            data = {"error": {"message": response.text}}

        if response.status_code != 200:
            message = data.get("error", {}).get("message", "YouTube API request failed.")
            raise ValueError(message)

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            top_comment = snippet.get("topLevelComment", {})
            comment_snippet = top_comment.get("snippet", {})
            text = (comment_snippet.get("textDisplay") or "").strip()
            if text:
                comments.append(text)
            if len(comments) >= max_comments:
                break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return video_id, comments


@app.route("/")
def home():
    return render_template("index.html", labels=LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    scores = model.predict_scores([text])[0]
    summary = summarize_scores(scores)

    return jsonify({
        "mode": "single",
        "inputText": text,
        "labelScores": scores,
        "activeLabels": summary["activeLabels"],
        "overallToxic": summary["overallToxic"]
    })


@app.route("/analyze-comments", methods=["POST"])
def analyze_comments():
    data = request.get_json(force=True)
    comments = data.get("comments", [])

    if not isinstance(comments, list) or len(comments) == 0:
        return jsonify({"error": "comments must be a non-empty list"}), 400

    cleaned_comments = [str(c).strip() for c in comments if str(c).strip()]
    if not cleaned_comments:
        return jsonify({"error": "No valid comments after cleaning"}), 400

    per_comment_scores = model.predict_scores(cleaned_comments)
    label_percentages = aggregate_label_percentages(per_comment_scores, threshold=50.0)
    overall_toxic_percent = round(
        sum(1 for scores in per_comment_scores if any(v >= 50.0 for v in scores.values())) / len(per_comment_scores) * 100,
        2
    )

    results = []
    for comment, scores in zip(cleaned_comments, per_comment_scores):
        summary = summarize_scores(scores)
        results.append({
            "comment": comment,
            "overallToxic": summary["overallToxic"],
            "activeLabels": summary["activeLabels"],
            "labelScores": scores
        })

    return jsonify({
        "mode": "batch",
        "source": "manual_comments",
        "totalComments": len(cleaned_comments),
        "overallToxicPercent": overall_toxic_percent,
        "labelPercentages": label_percentages,
        "results": results
    })


@app.route("/analyze-youtube", methods=["POST"])
def analyze_youtube():
    data = request.get_json(force=True)
    video_url = (data.get("videoUrl") or "").strip()
    max_comments = int(data.get("maxComments", 500))

    if not video_url:
        return jsonify({"error": "videoUrl is required"}), 400

    if max_comments < 1 or max_comments > 500:
        return jsonify({"error": "maxComments must be between 1 and 500"}), 400

    try:
        video_id, comments = get_youtube_comments(video_url, max_comments=max_comments)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.RequestException:
        return jsonify({"error": "Could not connect to the YouTube API."}), 502

    if not comments:
        return jsonify({"error": "No comments were retrieved for this video."}), 404

    per_comment_scores = model.predict_scores(comments)
    label_percentages = aggregate_label_percentages(per_comment_scores, threshold=50.0)
    overall_toxic_percent = round(
        sum(1 for scores in per_comment_scores if any(v >= 50.0 for v in scores.values())) / len(per_comment_scores) * 100,
        2
    )

    sample_results = []
    for comment, scores in list(zip(comments, per_comment_scores))[:-1]:
        summary = summarize_scores(scores)
        sample_results.append({
            "comment": comment,
            "overallToxic": summary["overallToxic"],
            "activeLabels": summary["activeLabels"],
            "labelScores": scores
        })

    return jsonify({
        "mode": "youtube",
        "videoId": video_id,
        "totalComments": len(comments),
        "overallToxicPercent": overall_toxic_percent,
        "labelPercentages": label_percentages,
        "results": sample_results
    })


def parse_channel_input(channel_input):
    value = (channel_input or "").strip()
    if not value:
        return None, None, None

    if re.fullmatch(r"UC[A-Za-z0-9_-]{22}", value):
        return value, None, None

    try:
        parsed = urlparse(value)
    except Exception:
        return None, None, None

    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/")

    if "youtube.com" not in host and "youtu.be" not in host:
        return None, None, None

    if path.startswith("channel/"):
        parts = path.split("/")
        if len(parts) > 1:
            return parts[1], None, None

    if path.startswith("user/"):
        parts = path.split("/")
        if len(parts) > 1:
            return None, parts[1], None

    if path.startswith("@"):
        return None, None, path[1:]

    return None, None, None


def resolve_channel_id(channel_input):
    direct_channel_id, username, handle = parse_channel_input(channel_input)
    if direct_channel_id:
        return direct_channel_id

    if not YOUTUBE_API_KEY:
        raise ValueError("Missing YOUTUBE_API_KEY environment variable.")

    if username:
        params = {"part": "id", "forUsername": username, "key": YOUTUBE_API_KEY}
        response = requests.get("https://www.googleapis.com/youtube/v3/channels", params=params, timeout=25)
        data = response.json() if response.content else {}
        if response.status_code != 200:
            message = data.get("error", {}).get("message", "YouTube API request failed.")
            raise ValueError(message)
        items = data.get("items", [])
        return items[0].get("id") if items else None

    if handle:
        params = {"part": "id", "forHandle": handle, "key": YOUTUBE_API_KEY}
        response = requests.get("https://www.googleapis.com/youtube/v3/channels", params=params, timeout=25)
        data = response.json() if response.content else {}
        if response.status_code != 200:
            message = data.get("error", {}).get("message", "YouTube API request failed.")
            raise ValueError(message)
        items = data.get("items", [])
        return items[0].get("id") if items else None

    return None


def get_channel_latest_videos(channel_id, max_videos=5):
    videos = []
    next_page_token = None

    while len(videos) < max_videos:
        batch_size = min(50, max_videos - len(videos))
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "order": "date",
            "maxResults": batch_size,
            "key": YOUTUBE_API_KEY
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get("https://www.googleapis.com/youtube/v3/search", params=params, timeout=25)
        data = response.json() if response.content else {}
        if response.status_code != 200:
            message = data.get("error", {}).get("message", "YouTube API request failed.")
            raise ValueError(message)

        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            title = item.get("snippet", {}).get("title", "").strip()
            if video_id:
                videos.append({"videoId": video_id, "title": title})
                if len(videos) >= max_videos:
                    break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return videos

def get_channel_metadata(channel_id):
    params = {
        "part": "snippet,statistics",
        "id": channel_id,
        "key": YOUTUBE_API_KEY
    }
    response = requests.get("https://www.googleapis.com/youtube/v3/channels", params=params, timeout=25)
    data = response.json() if response.content else {}
    if response.status_code != 200:
        message = data.get("error", {}).get("message", "YouTube API request failed.")
        raise ValueError(message)
    items = data.get("items", [])
    if not items:
        return {}
    channel = items[0]
    snippet = channel.get("snippet", {})
    statistics = channel.get("statistics", {})
    return {
        "channelId": channel_id,
        "title": snippet.get("title"),
        "subscriberCount": int(statistics.get("subscriberCount", 0)),
        "videoCount": int(statistics.get("videoCount", 0)),
        "viewCount": int(statistics.get("viewCount", 0)),
    }

@app.route("/analyze-youtube-channel", methods=["POST"])
def analyze_youtube_channel():
    data = request.get_json(force=True)
    channel_input = (data.get("channelUrl") or "").strip()
    max_videos = int(data.get("maxVideos", 5))
    max_comments_per_video = int(data.get("maxCommentsPerVideo", 50))

    if not channel_input:
        return jsonify({"error": "channelUrl is required"}), 400

    if max_videos < 1 or max_videos > 20:
        return jsonify({"error": "maxVideos must be between 1 and 20"}), 400

    if max_comments_per_video < 1 or max_comments_per_video > 500:
        return jsonify({"error": "maxCommentsPerVideo must be between 1 and 500"}), 400

    try:
        channel_id = resolve_channel_id(channel_input)
        if not channel_id:
            return jsonify({"error": "Could not resolve channel ID from the provided input."}), 400

        channel_meta = get_channel_metadata(channel_id)
        latest_videos = get_channel_latest_videos(channel_id, max_videos=max_videos)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.RequestException:
        return jsonify({"error": "Could not connect to the YouTube API."}), 502

    if not latest_videos:
        return jsonify({"error": "No recent videos were found for this channel."}), 404

    channel_comments = []
    per_video_results = []

    for video in latest_videos:
        video_url = f"https://www.youtube.com/watch?v={video['videoId']}"
        try:
            _, comments = get_youtube_comments(video_url, max_comments=max_comments_per_video)
        except Exception:
            comments = []

        if comments:
            per_comment_scores = model.predict_scores(comments)
            label_percentages = aggregate_label_percentages(per_comment_scores, threshold=50.0)
            overall_toxic_percent = round(
                sum(1 for scores in per_comment_scores if any(v >= 50.0 for v in scores.values())) / len(per_comment_scores) * 100,
                2
            )
            sample_comments = []
            for comment, scores in zip(comments, per_comment_scores):
                summary = summarize_scores(scores)
                sample_comments.append({
                    "comment": comment,
                    "overallToxic": summary["overallToxic"],
                    "activeLabels": summary["activeLabels"],
                    "labelScores": scores
                })
        else:
            label_percentages = {label: 0.0 for label in LABELS}
            overall_toxic_percent = 0.0
            sample_comments = []

        per_video_results.append({
            "videoId": video["videoId"],
            "title": video["title"],
            "totalComments": len(comments),
            "overallToxicPercent": overall_toxic_percent,
            "labelPercentages": label_percentages,
            "results": sample_comments
        })
        channel_comments.extend(comments)

    if channel_comments:
        channel_scores = model.predict_scores(channel_comments)
        channel_label_percentages = aggregate_label_percentages(channel_scores, threshold=50.0)
        channel_overall_toxic_percent = round(
            sum(1 for scores in channel_scores if any(v >= 50.0 for v in scores.values())) / len(channel_scores) * 100,
            2
        )
    else:
        channel_label_percentages = {label: 0.0 for label in LABELS}
        channel_overall_toxic_percent = 0.0

    return jsonify({
        "mode": "youtube_channel",
        "channel": channel_meta,
        "videosRequested": max_videos,
        "commentsRequestedPerVideo": max_comments_per_video,
        "videosScanned": len(per_video_results),
        "totalCommentsScanned": len(channel_comments),
        "overallToxicPercent": channel_overall_toxic_percent,
        "labelPercentages": channel_label_percentages,
        "videos": per_video_results
    })


if __name__ == "__main__":
    app.run(debug=True)