// Troque pelo endpoint real do Render, ex.:
// const API_URL = "https://ai-automation-risk-api.onrender.com/predict";
const API_URL = "https://ai-impact-on-job-market-2024-2030-api.onrender.com/predict";

const form = document.getElementById("risk-form");
const resultBox = document.getElementById("result");
const riskValueSpan = document.getElementById("risk-value");
const errorBox = document.getElementById("error");
const submitBtn = document.getElementById("submit-btn");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");

  const payload = {
    Industry: document.getElementById("industry").value,
    Job_Status: document.getElementById("jobStatus").value,
    AI_Impact_Level: document.getElementById("aiImpact").value,
    Required_Education: document.getElementById("education").value,
    Median_Salary_USD: Number(document.getElementById("salary").value),
    Experience_Required_Years: Number(document.getElementById("experience").value),
    Job_Openings_2024: Number(document.getElementById("openings2024").value),
    Projected_Openings_2030: Number(document.getElementById("openings2030").value),
    Remote_Work_Ratio: Number(document.getElementById("remoteRatio").value),
    Gender_Diversity: Number(document.getElementById("genderDiversity").value),
  };

  // Validação básica
  if (!payload.Industry || !payload.Job_Status || !payload.AI_Impact_Level || !payload.Required_Education) {
    showError("Preencha todos os campos obrigatórios.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Calculando...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Erro ao chamar a API.");
    }

    const data = await response.json();
    const risk = data.automation_risk_predicted;

    riskValueSpan.textContent = risk.toFixed(2);
    resultBox.classList.remove("hidden");
  } catch (err) {
    console.error(err);
    showError(err.message || "Erro inesperado ao calcular o risco.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Calcular risco de automação";
  }
});

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}
