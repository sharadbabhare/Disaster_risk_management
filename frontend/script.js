const btn = document.getElementById("btn");
const statusBox = document.getElementById("status");
const cardsBox = document.getElementById("cards");

const BACKEND = "http://127.0.0.1:8001/predict-live";

function getBadgeClass(prob) {
  if (prob < 30) return ["SAFE", "safe"];
  if (prob < 60) return ["WARNING", "warning"];
  return ["DANGER", "danger"];
}

function makeCard(title, emoji, probPercent, inputData) {
  const [label, cls] = getBadgeClass(probPercent);

  return `
    <div class="card">
      <h2>${emoji} ${title}</h2>
      <span class="badge ${cls}">${label}</span>
      <div class="prob">${probPercent}%</div>

      <details>
        <summary>Inputs Used</summary>
        <pre>${JSON.stringify(inputData, null, 2)}</pre>
      </details>
    </div>
  `;
}

btn.addEventListener("click", async () => {
  statusBox.textContent = "📍 Getting location...";
  cardsBox.classList.add("hidden");
  cardsBox.innerHTML = "";

  navigator.geolocation.getCurrentPosition(async (pos) => {
    const lat = pos.coords.latitude;
    const lon = pos.coords.longitude;

    statusBox.textContent = `✅ Location: ${lat}, ${lon}\n🔄 Calling backend...`;

    try {
      const res = await fetch(`${BACKEND}?lat=${lat}&lon=${lon}`);
      const data = await res.json();

      statusBox.textContent = `✅ Prediction Done Successfully`;

      // Cards
      const heatwaveCard = makeCard(
        "Heatwave",
        "🌡️",
        data.heatwave.probability__percent,
        data.heatwave_features_used
      );

      const landslideCard = makeCard(
        "Landslide",
        "⛰️",
        data.landslide.probability__percent,
        data.landslide_features_used
      );

      // Flood (if backend sends it)
      let floodCard = "";
      if (data.flood) {
        floodCard = makeCard(
          "Flood",
          "🌊",
          data.flood.probability__percent,
          data.flood_features_used
        );
      } else {
        floodCard = makeCard(
          "Flood",
          "🌊",
          0,
          { note: "Flood model not connected in backend yet" }
        );
      }

      cardsBox.innerHTML = heatwaveCard + landslideCard + floodCard;
      cardsBox.classList.remove("hidden");

    } catch (err) {
      statusBox.textContent = "❌ Error: " + err;
    }
  }, () => {
    statusBox.textContent = "❌ Location permission denied.";
  });
});
