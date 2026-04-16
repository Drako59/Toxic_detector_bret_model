const tabs = document.querySelectorAll(".tab");
const panels = {
    single: document.getElementById("single-panel"),
    batch: document.getElementById("batch-panel"),
    youtube: document.getElementById("youtube-panel")
};

tabs.forEach(tab => {
    tab.addEventListener("click", () => {
        tabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const chosen = tab.dataset.tab;
        Object.entries(panels).forEach(([key, panel]) => {
            panel.classList.toggle("active", key === chosen);
        });
    });
});

document.getElementById("singleBtn").addEventListener("click", analyzeSingle);
document.getElementById("batchBtn").addEventListener("click", analyzeBatch);
document.getElementById("youtubeBtn").addEventListener("click", analyzeYoutube);

async function analyzeSingle() {
    const text = document.getElementById("singleText").value.trim();
    const resultBox = document.getElementById("singleResult");

    if (!text) {
        resultBox.classList.remove("hidden");
        resultBox.innerHTML = "Please enter text first.";
        return;
    }

    resultBox.classList.remove("hidden");
    resultBox.innerHTML = "Analyzing...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ text })
        });

        const data = await response.json();
        if (!response.ok) {
            resultBox.innerHTML = escapeHtml(data.error || "Something went wrong.");
            return;
        }

        resultBox.innerHTML = `
            <div><strong>Input:</strong> ${escapeHtml(data.inputText)}</div>
            <div class="top-gap">
                ${data.overallToxic
                    ? '<span class="badge toxic">Overall: Toxic</span>'
                    : '<span class="badge safe">Overall: Not Toxic</span>'}
            </div>
            <div class="top-gap"><strong>Triggered labels:</strong> ${renderTriggeredLabels(data.activeLabels)}</div>
            <div class="labels-grid top-gap">
                ${renderLabelCards(data.labelScores)}
            </div>
        `;
    } catch (error) {
        resultBox.innerHTML = "Could not connect to backend.";
    }
}

async function analyzeBatch() {
    const raw = document.getElementById("batchComments").value;
    const comments = raw.split("\n").map(x => x.trim()).filter(Boolean);

    const summary = document.getElementById("batchSummary");
    const tableWrap = document.getElementById("batchTableWrap");
    const tableBody = document.getElementById("batchTableBody");

    if (comments.length === 0) {
        summary.classList.remove("hidden");
        summary.innerHTML = "Please enter at least one comment.";
        tableWrap.classList.add("hidden");
        return;
    }

    summary.classList.remove("hidden");
    summary.innerHTML = "Analyzing comments...";
    tableWrap.classList.add("hidden");
    tableBody.innerHTML = "";

    try {
        const response = await fetch("/analyze-comments", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ comments })
        });

        const data = await response.json();
        if (!response.ok) {
            summary.innerHTML = escapeHtml(data.error || "Something went wrong.");
            return;
        }

        summary.innerHTML = `
            <h3>Summary</h3>
            <div class="summary-grid">
                <div class="metric">Total Comments<strong>${data.totalComments}</strong></div>
                <div class="metric">Overall Toxic %<strong>${data.overallToxicPercent}%</strong></div>
            </div>
            <div class="labels-grid top-gap">
                ${renderLabelCards(data.labelPercentages, true)}
            </div>
        `;

        data.results.forEach((item, index) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td>${escapeHtml(item.comment)}</td>
                <td>${renderTriggeredLabels(item.activeLabels)}</td>
            `;
            tableBody.appendChild(tr);
        });

        tableWrap.classList.remove("hidden");
    } catch (error) {
        summary.innerHTML = "Could not connect to backend.";
    }
}

async function analyzeYoutube() {
    const videoUrl = document.getElementById("youtubeUrl").value.trim();
    const maxComments = parseInt(document.getElementById("youtubeMaxComments").value, 10) || 100;

    const summary = document.getElementById("youtubeSummary");
    const labelsWrap = document.getElementById("youtubeLabels");
    const tableWrap = document.getElementById("youtubeTableWrap");
    const tableBody = document.getElementById("youtubeTableBody");

    if (!videoUrl) {
        summary.classList.remove("hidden");
        summary.innerHTML = "Please enter a YouTube video URL.";
        labelsWrap.classList.add("hidden");
        tableWrap.classList.add("hidden");
        return;
    }

    summary.classList.remove("hidden");
    summary.innerHTML = "Fetching YouTube comments and analyzing them...";
    labelsWrap.classList.add("hidden");
    tableWrap.classList.add("hidden");
    tableBody.innerHTML = "";

    try {
        const response = await fetch("/analyze-youtube", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ videoUrl, maxComments })
        });

        const data = await response.json();
        if (!response.ok) {
            summary.innerHTML = escapeHtml(data.error || "Something went wrong.");
            return;
        }

        summary.innerHTML = `
            <h3>YouTube Scan Summary</h3>
            <div class="summary-grid">
                <div class="metric">Video ID<strong>${escapeHtml(data.videoId)}</strong></div>
                <div class="metric">Comments Scanned<strong>${data.totalComments}</strong></div>
                <div class="metric">Overall Toxic %<strong>${data.overallToxicPercent}%</strong></div>
            </div>
        `;

        labelsWrap.innerHTML = renderLabelCards(data.labelPercentages, true);
        labelsWrap.classList.remove("hidden");

        data.results.forEach((item, index) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td>${escapeHtml(item.comment)}</td>
                <td>${renderTriggeredLabels(item.activeLabels)}</td>
            `;
            tableBody.appendChild(tr);
        });

        tableWrap.classList.remove("hidden");
    } catch (error) {
        summary.innerHTML = "Could not connect to backend.";
    }
}

function renderTriggeredLabels(labels) {
    if (!labels || labels.length === 0) {
        return '<span class="badge safe">None</span>';
    }
    return labels.map(label => `<span class="badge toxic">${escapeHtml(label)}</span>`).join("");
}

function renderLabelCards(labelMap, asPercentSummary = false) {
    const labels = window.ALL_LABELS || Object.keys(labelMap);
    return labels.map(label => {
        const value = Number(labelMap[label] ?? 0).toFixed(2);
        const title = escapeHtml(label);
        return `
            <div class="label-card">
                <div class="muted">${title}</div>
                <strong>${value}%</strong>
                <div class="progress"><span style="width:${Math.max(0, Math.min(100, value))}%"></span></div>
            </div>
        `;
    }).join("");
}

function escapeHtml(text) {
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}