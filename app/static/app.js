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
      <input class="weight-number" type="number" id="n-${key}" min="0" max="100" value="${value}">
      <b class="weight-value" id="v-${key}">%</b>
    </div>`);
});

let adjustingWeights = false;
let allRecommendations = [];
let currentPage = 1;
const pageSize = 10;

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
    document.querySelector(`#n-${key}`).value = value;
    document.querySelector(`#v-${key}`).textContent = "%";
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
  document.querySelector(`#n-${key}`).addEventListener("change", event => {
    rebalanceWeights(key, event.target.value);
  });
});
updateWeightDisplay();

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    if (!response.ok) throw new Error("model-info unavailable");
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

async function readErrorMessage(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      const data = await response.json();
      return data.detail || data.message || "后端接口暂时不可用";
    } catch {
      return "后端接口暂时不可用";
    }
  }
  if (response.status === 502 || response.status === 503 || response.status === 504) {
    return "云端服务正在重启或内存不足，请稍等 1 分钟后重试；如果反复出现，需要检查 Render 部署状态。";
  }
  const text = await response.text();
  const compact = text.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  return compact ? compact.slice(0, 240) : `请求失败，HTTP ${response.status}`;
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><b>${value}</b></div>`;
}

function renderCard(item, index) {
  const p = item.properties;
  const components = item.components || [
    {code: item.solvent_a, name: item.solvent_a_name, ratio: item.ratio_a},
    {code: item.solvent_b, name: item.solvent_b_name, ratio: item.ratio_b},
  ];
  const formula = components.map(component => component.code).join(" + ");
  const names = components.map(component => component.name).join(" / ");
  const ratioSegments = components
    .map(component => `<i style="width:${component.ratio}%"></i>`)
    .join("");
  const ratioLabels = components
    .map(component => `<span>${component.code} ${component.ratio}%</span>`)
    .join("");
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
          <h3>${formula}</h3>
          <div class="names">${names}</div>
          <div class="ratio-bar">${ratioSegments}</div>
          <div class="ratio-labels">${ratioLabels}</div>
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

function renderPage() {
  const cards = document.querySelector("#cards");
  const pager = document.querySelector("#pager");
  const pageInfo = document.querySelector("#page-info");
  const totalPages = Math.max(1, Math.ceil(allRecommendations.length / pageSize));
  currentPage = Math.min(Math.max(currentPage, 1), totalPages);
  const start = (currentPage - 1) * pageSize;
  const pageItems = allRecommendations.slice(start, start + pageSize);
  cards.innerHTML = pageItems.map((item, index) => renderCard(item, start + index)).join("");
  pageInfo.textContent = `第 ${currentPage} / ${totalPages} 页 · 共 ${allRecommendations.length} 个候选`;
  document.querySelector("#prev-page").disabled = currentPage <= 1;
  document.querySelector("#next-page").disabled = currentPage >= totalPages;
  pager.classList.toggle("hidden", allRecommendations.length <= pageSize);
}

async function runScreening() {
  const button = document.querySelector("#run");
  const loading = document.querySelector("#loading");
  const empty = document.querySelector("#empty");
  const cards = document.querySelector("#cards");
  const notice = document.querySelector("#notice");
  const pager = document.querySelector("#pager");
  button.disabled = true;
  empty.classList.add("hidden");
  cards.innerHTML = "";
  allRecommendations = [];
  currentPage = 1;
  pager.classList.add("hidden");
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
    score_threshold: Number(document.querySelector("#score-threshold").value),
    max_results: 120,
    max_components: Number(document.querySelector("#max-components").value),
    return_all_above_threshold: true,
    allow_relaxed_fallback: true,
    weights,
  };

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await readErrorMessage(response));
    const data = await response.json();
    notice.textContent = data.warning;
    notice.classList.remove("hidden");
    document.querySelector("#search-stat").textContent =
      `${data.search_space.solvents} 种溶剂 · ${data.search_space.evaluated_formulations.toLocaleString()} 个候选 · 返回 ${data.recommendations.length} 个`;
    allRecommendations = data.recommendations;
    renderPage();
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
document.querySelector("#prev-page").addEventListener("click", () => {
  currentPage -= 1;
  renderPage();
});
document.querySelector("#next-page").addEventListener("click", () => {
  currentPage += 1;
  renderPage();
});
loadModelInfo();

const moleculeStage = document.querySelector(".molecule-stage");
if (moleculeStage) {
  document.querySelector(".hero").addEventListener("pointermove", event => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width - 0.5;
    const y = (event.clientY - rect.top) / rect.height - 0.5;
    moleculeStage.style.setProperty("--tilt-x", `${x * 8}deg`);
    moleculeStage.style.setProperty("--tilt-y", `${-y * 5}deg`);
  });
}
