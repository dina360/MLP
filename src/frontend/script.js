
document.addEventListener("DOMContentLoaded", () => {
  const detectBtn = document.getElementById("detectBtn");
  const inputText = document.getElementById("inputText");
  const resultDiv = document.getElementById("result");
  const predictedLangP = document.getElementById("predictedLang");

  detectBtn.addEventListener("click", () => {
    const text = inputText.value.trim();
    if (!text) {
      alert("Veuillez entrer un texte avant de cliquer.");
      return;
    }

    fetch("/classify", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
      },
      body: new URLSearchParams({ text: text })
    })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Problème réseau ou serveur indisponible");
      }
      return response.json();
    })
    .then((data) => {
      if (data.language) {
        predictedLangP.textContent = data.language;
        resultDiv.classList.remove("hidden");
      } else {
        predictedLangP.textContent = "Erreur : aucune langue détectée";
        resultDiv.classList.remove("hidden");
      }
    })
    .catch((err) => {
      alert("Erreur : " + err.message);
    });
  });
});
