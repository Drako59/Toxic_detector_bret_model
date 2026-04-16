const tabs = document.querySelectorAll(".tab");
const singlePanel = document.getElementById("single-panel");
const batchPanel = document.getElementById("batch-panel");

tabs.forEach(tab => {
    tab.addEventListener("click", () => {
        tabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        const chosen = tab.dataset.tab;
        singlePanel.classList.toggle("active", chosen === "single");
        batchPanel.classList.toggle("active", chosen === "batch");
    });
});

document.getElementById("singleBtn").addEventListener("click", analyzeSingle);
document.getElementById("batchBtn").addEventListener("click", analyzeBatch);

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
            resultBox.innerHTML = data.error || "Something went wrong.";
            return;
        }

        const badgeClass = data.prediction === 1 ? "toxic" : "safe";

        resultBox.innerHTML = `
            <div><strong>Input:</strong> ${escapeHtml(data.inputText)}</div>
            <div style="margin-top:10px;">
                <span class="badge ${badgeClass}">${data.label}</span>
            </div>
        `;
    } catch (error) {
        resultBox.innerHTML = "Could not connect to backend.";
    }
}

async function analyzeBatch() {
    const raw = document.getElementById("batchComments").value;
    const comments = raw
        .split("\n")
        .map(x => x.trim())
        .filter(Boolean);

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
            summary.innerHTML = data.error || "Something went wrong.";
            return;
        }

        summary.innerHTML = `
            <h3>Summary</h3>
            <div class="summary-grid">
                <div class="metric">Total Comments<strong>${data.totalComments}</strong></div>
                <div class="metric">Toxic Comments<strong>${data.toxicComments}</strong></div>
                <div class="metric">Toxicity Percent<strong>${data.toxicityPercent}%</strong></div>
                <div class="metric">Recommendation<strong>${escapeHtml(data.recommendation)}</strong></div>
            </div>
        `;

        data.results.forEach((item, index) => {
            const tr = document.createElement("tr");
            const badgeClass = item.prediction === 1 ? "toxic" : "safe";
            tr.innerHTML = `
                <td>${index + 1}</td>
                <td>${escapeHtml(item.comment)}</td>
                <td><span class="badge ${badgeClass}">${item.label}</span></td>
            `;
            tableBody.appendChild(tr);
        });

        tableWrap.classList.remove("hidden");
    } catch (error) {
        summary.innerHTML = "Could not connect to backend.";
    }
}

function escapeHtml(text) {
    return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
