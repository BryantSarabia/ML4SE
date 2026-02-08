const API_URL = "http://localhost:8000";

let exampleComments = [];

async function loadExamples() {
  try {
    const response = await fetch(`${API_URL}/examples`);
    if (response.ok) {
      exampleComments = await response.json();
    }
  } catch (error) {
    console.log("Could not load examples, using fallbacks");
    exampleComments = [
      { text: "This is a great article! Very informative.", category: "clean" },
      { text: "You are stupid and don't know anything!", category: "toxic" },
      { text: "An article you've written has been nominated for deletion.  Please see Wikipedia:Articles for deletion/Maude, I Swear I Shall Fucking Kill That Flea-bitten Cur! if you'd like to contribute to the discussion.", category: "threat" },
      { text: "What an idiot, go back to school!", category: "insult" },
    ];
  }
}

function loadExample(index) {
  if (exampleComments[index]) {
    document.getElementById("comment-input").value =
      exampleComments[index].text;
  }
}

async function analyzeComment() {
  const commentInput = document.getElementById("comment-input");
  const comment = commentInput.value.trim();

  if (!comment || comment.length < 3) {
    showError("Please enter a valid comment (at least 3 characters)");
    commentInput.focus();
    return;
  }

  hideError();
  showLoading();

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ comment: comment }),
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const data = await response.json();
    displayResults(data);
  } catch (error) {
    showError(
      "Error: Could not connect to the API server. Make sure the server is running.",
    );
    hideResults();
  } finally {
    hideLoading();
  }
}

function displayResults(data) {
  const predictions = data.predictions;
  const riskLevel = data.risk_level;
  const highestRisk = data.highest_risk;

  document.getElementById("risk-level").textContent = riskLevel.toUpperCase();
  document.getElementById("risk-level").className = `risk-badge ${riskLevel}`;

  for (const [label, probability] of Object.entries(predictions)) {
    const percentage = (probability * 100).toFixed(1);
    const barElement = document.getElementById(`${label}-bar`);
    const pctElement = document.getElementById(`${label}-pct`);

    if (barElement && pctElement) {
      barElement.style.width = `${percentage}%`;
      pctElement.textContent = `${percentage}%`;
    }
  }

  document.getElementById("highest-risk-label").textContent =
    highestRisk.replace("_", " ");
  document.getElementById("results-section").style.display = "block";
}

function showLoading() {
  document.getElementById("analyze-btn").textContent = "Analyzing...";
  document.getElementById("analyze-btn").disabled = true;
}

function hideLoading() {
  document.getElementById("analyze-btn").textContent = "Analyze Comment";
  document.getElementById("analyze-btn").disabled = false;
}

function showError(message) {
  const errorEl = document.getElementById("error-message");
  errorEl.textContent = message;
  errorEl.style.display = "block";
  hideLoading();
}

function hideError() {
  document.getElementById("error-message").style.display = "none";
}

function hideResults() {
  document.getElementById("results-section").style.display = "none";
}

document
  .getElementById("comment-input")
  .addEventListener("keypress", function (e) {
    if (e.key === "Enter" && e.ctrlKey) {
      analyzeComment();
    }
  });

window.addEventListener("load", () => {
  loadExamples();
  hideLoading();
});
