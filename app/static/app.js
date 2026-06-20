const weightConfig = [
  ["solubility", "溶解能力", 35],
  ["conductivity", "离子传输", 30],
  ["stability", "电化学稳定", 15],
  ["safety", "安全性", 12],
  ["low_temperature", "低温表现", 8],
];

const weightContainer = document.querySelector("#weights");
weightConfig.forEach(([key, label, value]) => {
  weightContainer.insertAdjacentHTML("beforeend", `
    <div class="weight-row">
      <span>${label}</span>
      <input type="range" id="w-${key}" min="0" max="100" value="${value}">
      <b class="weight-value" id="v-${key}">${value}</b>
    </div>`);
});

let adjustingWeights = false;

function largestRemainder(values, targetTotal) {
  const floors = values.map(Math.floor);
  let remainder = targetTotal - floors.reduce((sum, value) => sum + value, 0);
  const order = values
    .map((value, index) => ({index, fraction: value - floors[index]}))
    .sort((a, b) => b.fraction - a.fraction);
  for (let i = 0; i < remainder; i += 1) floors[order[i].index] += 1;
  return floors;
}

function updateWeightDisplay() {
  let total = 0;
  weightConfig.forEach(([key]) => {
    const value = Number(document.querySelector(`#w-${key}`).value);
    total += value;
    document.querySelector(`#v-${key}`).textContent = value;
  });
  const totalBadge = document.querySelector("#weight-total");
  totalBadge.textContent = `总权重 ${total}%`;
  totalBadge.classList.toggle("invalid", total !== 100);
}

function rebalanceWeights(changedKey, requestedValue) {
  if (adjustingWeights) return;
  adjustingWeights = true;
  const keys = weightConfig.map(([key]) => key);
  const changedIndex = keys.indexOf(changedKey);
  const upperKeys = keys.slice(0, changedIndex);
  const lowerKeys = keys.slice(changedIndex + 1);
  const upperTotal = upperKeys.reduce(
    (sum, key) => sum + Number(document.querySelector(`#w-${key}`).value),
    0,
  );
  const maximumCurrent = Math.max(0, 100 - upperTotal);

  // The final slider represents the exact remainder after all choices above it.
  // Dragging it therefore snaps back to the available remainder.
  const changedValue = lowerKeys.length === 0
    ? maximumCurrent
    : Math.max(0, Math.min(maximumCurrent, Number(requestedValue)));
  document.querySelector(`#w-${changedKey}`).value = changedValue;

  const remaining = maximumCurrent - changedValue;
  if (lowerKeys.length > 0) {
    const lowerValues = lowerKeys.map(
      key => Number(document.querySelector(`#w-${key}`).value),
    );
    const lowerTotal = lowerValues.reduce((sum, value) => sum + value, 0);
    const rawValues = lowerTotal > 0
      ? lowerValues.map(value => value / lowerTotal * remaining)
      : lowerValues.map(() => remaining / lowerValues.length);
    const balanced = largestRemainder(rawValues, remaining);
    lowerKeys.forEach((key, index) => {
      document.querySelector(`#w-${key}`).value = balanced[index];
    });
  }
  updateWeightDisplay();
  adjustingWeights = false;
}

weightConfig.forEach(([key]) => {
  document.querySelector(`#w-${key}`).addEventListener("input", event => {
    rebalanceWeights(key, event.target.value);
  });
});
updateWeightDisplay();

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    const info = await response.json();
    const pill = document.querySelector("#model-pill");
    const lino3Rows = info.lino3_solubility_model?.metrics?.rows || 0;
    pill.textContent = info.available
      ? `模型在线 · 电导率 ${info.metrics.train_rows.toLocaleString()} 条 · LiNO₃ 溶解度 ${lino3Rows} 条`
      : "启发式模式 · 模型尚未训练";
  } catch {
    document.querySelector("#model-pill").textContent = "服务连接异常";
  }
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><b>${value}</b></div>`;
}

function renderCard(item, index) {
  const p = item.properties;
  const reasons = item.reasons.map(x => `<span class="reason">${x}</span>`).join("");
  const violations = (item.constraint_violations || [])
    .map(x => `<span class="reason violation">${x}</span>`).join("");
  const factorLabels = {
    domain_similarity: "训练域相似",
    ensemble_agreement: "模型一致",
    local_data_density: "局部数据",
    temperature_coverage: "温度覆盖",
    component_coverage: "溶剂覆盖",
    physics_model_agreement: "机理一致",
  };
  const confidenceFactors = Object.entries(item.confidence_factors || {})
    .map(([key, value]) => `<span class="confidence-factor">${factorLabels[key] || key} ${Math.round(value * 100)}%</span>`)
    .join("");
  const conductivity = item.predicted_conductivity === null ? "代理分数" : item.predicted_conductivity;
  const conductivityLabel = item.predicted_conductivity === null ? "传输模式" : "预测电导率";
  const solubilityValue = item.predicted_solubility_mole_fraction === null
    ? p.solubility_score
    : item.predicted_solubility_mole_fraction;
  const solubilityLabel = item.predicted_solubility_mole_fraction === null
    ? "溶解评分"
    : "预测溶解度 x";
  return `
    <article class="result-card" style="animation-delay:${index * 35}ms">
      <div class="card-main">
        <div class="rank">${String(index + 1).padStart(2, "0")}</div>
        <div class="formula">
          <h3>${item.solvent_a} <span>+</span> ${item.solvent_b}</h3>
          <div class="names">${item.solvent_a_name} / ${item.solvent_b_name}</div>
          <div class="ratio-bar"><i style="width:${item.ratio_a}%"></i></div>
          <div class="ratio-labels"><span>${item.solvent_a} ${item.ratio_a}%</span><span>${item.solvent_b} ${item.ratio_b}%</span></div>
        </div>
        <div class="score-group">
          <div class="score-ring" style="--score:${item.score}%"><b>${item.score}</b></div>
          <div class="metrics">
            ${metric("置信度", `${item.confidence}%`)}
            ${metric(conductivityLabel, conductivity)}
            ${metric("混合黏度", `${p.viscosity_mpas} mPa·s`)}
            ${metric(solubilityLabel, solubilityValue)}
            ${metric("稳定评分", p.stability_score)}
            ${metric("估算闪点", `${p.flash_point_c} °C`)}
          </div>
        </div>
      </div>
      <div class="card-detail">
        <span class="basis">${item.basis}</span>
        <div class="reasons">${confidenceFactors}${reasons}${violations}</div>
      </div>
    </article>`;
}

async function runScreening() {
  const button = document.querySelector("#run");
  const loading = document.querySelector("#loading");
  const empty = document.querySelector("#empty");
  const cards = document.querySelector("#cards");
  const notice = document.querySelector("#notice");
  button.disabled = true;
  empty.classList.add("hidden");
  cards.innerHTML = "";
  notice.classList.add("hidden");
  loading.classList.remove("hidden");

  const weights = {};
  weightConfig.forEach(([key]) => weights[key] = Number(document.querySelector(`#w-${key}`).value) / 100);
  const body = {
    salt: document.querySelector("#salt").value,
    temperature_c: Number(document.querySelector("#temperature").value),
    concentration: Number(document.querySelector("#concentration").value),
    concentration_unit: "mol/kg",
    application: document.querySelector("#application").value,
    min_flash_point_c: Number(document.querySelector("#flash").value),
    max_mixture_viscosity: Number(document.querySelector("#viscosity").value),
    exclude_high_hazard: document.querySelector("#hazard").checked,
    top_k: 10,
    allow_relaxed_fallback: true,
    weights,
  };

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    notice.textContent = data.warning;
    notice.classList.remove("hidden");
    document.querySelector("#search-stat").textContent =
      `${data.search_space.solvents} 种溶剂 · ${data.search_space.evaluated_formulations.toLocaleString()} 个可行配方`;
    cards.innerHTML = data.recommendations.map(renderCard).join("");
    if (!data.recommendations.length) {
      empty.querySelector("h3").textContent = "当前约束下没有可行结果";
      empty.querySelector("p").textContent = "可适当降低最低闪点或提高最高黏度限制。";
      empty.classList.remove("hidden");
    }
  } catch (error) {
    notice.textContent = `运行失败：${error.message}`;
    notice.classList.remove("hidden");
    empty.classList.remove("hidden");
  } finally {
    loading.classList.add("hidden");
    button.disabled = false;
  }
}

document.querySelector("#run").addEventListener("click", runScreening);
loadModelInfo();
