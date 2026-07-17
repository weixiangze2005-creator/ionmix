const weightConfig = [
  ["solubility", "溶解能力", 35],
  ["conductivity", "离子传输", 30],
  ["stability", "电化学稳定", 15],
  ["safety", "安全性", 12],
  ["low_temperature", "低温表现", 8],
];

const weightPresets = {
  balanced: [30, 30, 15, 15, 10],
  solubility: [55, 20, 10, 10, 5],
  transport: [20, 55, 10, 10, 5],
  safety: [20, 15, 10, 45, 10],
};
const settingsKey = "ionmix-screening-settings-v2";

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
let currentUser = null;
let lastRequestBody = null;
let activeRequestController = null;
let loadingTimer = null;

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

function setWeights(values) {
  weightConfig.forEach(([key], index) => {
    document.querySelector(`#w-${key}`).value = values[index];
  });
  updateWeightDisplay();
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

document.querySelectorAll("[data-preset]").forEach(button => {
  button.addEventListener("click", () => {
    setWeights(weightPresets[button.dataset.preset]);
    document.querySelectorAll("[data-preset]").forEach(item => item.classList.remove("active"));
    button.classList.add("active");
    saveSettings();
  });
});

function saveSettings() {
  const values = {
    salt: document.querySelector("#salt").value,
    temperature: document.querySelector("#temperature").value,
    concentration: document.querySelector("#concentration").value,
    application: document.querySelector("#application").value,
    maxComponents: document.querySelector("#max-components").value,
    scoreThreshold: document.querySelector("#score-threshold").value,
    flash: document.querySelector("#flash").value,
    viscosity: document.querySelector("#viscosity").value,
    hazard: document.querySelector("#hazard").checked,
    weights: weightConfig.map(([key]) => Number(document.querySelector(`#w-${key}`).value)),
  };
  localStorage.setItem(settingsKey, JSON.stringify(values));
}

function restoreSettings() {
  try {
    const values = JSON.parse(localStorage.getItem(settingsKey) || "null");
    if (!values) return;
    const fields = {
      salt: "salt",
      temperature: "temperature",
      concentration: "concentration",
      application: "application",
      maxComponents: "max-components",
      scoreThreshold: "score-threshold",
      flash: "flash",
      viscosity: "viscosity",
    };
    Object.entries(fields).forEach(([key, id]) => {
      if (values[key] !== undefined) document.querySelector(`#${id}`).value = values[key];
    });
    if (typeof values.hazard === "boolean") document.querySelector("#hazard").checked = values.hazard;
    if (Array.isArray(values.weights) && values.weights.length === weightConfig.length
      && values.weights.reduce((sum, value) => sum + Number(value), 0) === 100) {
      setWeights(values.weights);
    }
  } catch {
    localStorage.removeItem(settingsKey);
  }
}

restoreSettings();

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model-info");
    if (!response.ok) throw new Error("model-info unavailable");
    const info = await response.json();
    const pill = document.querySelector("#model-pill");
    const lino3Rows = info.lino3_solubility_model?.metrics?.rows || 0;
    const mixture = info.mixture_property_model;
    const mixtureRows = mixture?.metrics?.training_summary?.rows || 0;
    const lino3BinaryRows = mixture?.metrics?.solubility?.lino3_binary_rows || 0;
    const oedb = info.oedb_auxiliary_model;
    const oedbRows = oedb?.metrics?.training_summary?.rows || 0;
    const solventCount = info.solvent_catalog?.count || 0;
    const account = info.account_storage || {};
    const accountText = account.available
      ? (account.backend === "postgresql" ? "云端账户" : "本地账户")
      : "账户待配置";
    pill.classList.toggle("degraded", !account.available);
    if (mixture?.available) {
      pill.textContent = oedb?.available
        ? `模型在线 · ${mixtureRows.toLocaleString()} 实验 · ${oedbRows.toLocaleString()} 模拟 · ${solventCount} 溶剂 · ${accountText}`
        : `模型在线 · ${mixtureRows.toLocaleString()} 实验 · ${solventCount} 溶剂 · ${accountText}`;
      pill.title = `公开实验 ${mixtureRows.toLocaleString()} 条；OEDB-MD ${oedbRows.toLocaleString()} 条；LiNO₃ 二元标签 ${lino3BinaryRows} 条；候选溶剂 ${solventCount} 种。`;
    } else if (info.available) {
      pill.textContent = `模型在线 · 电导率 ${info.metrics.train_rows.toLocaleString()} 条 · LiNO₃ 溶解度 ${lino3Rows} 条 · ${accountText}`;
    } else if (lino3Rows > 0) {
      pill.textContent = `轻量模型在线 · LiNO₃ 溶解度 ${lino3Rows} 条 · ${accountText}`;
    } else {
      pill.textContent = "启发式模式 · 模型尚未训练";
    }
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

function openModal(id) {
  document.querySelector(`#${id}`).classList.remove("hidden");
}

function closeModal(id) {
  document.querySelector(`#${id}`).classList.add("hidden");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderAuthState() {
  const loginButton = document.querySelector("#auth-open");
  const historyButton = document.querySelector("#history-open");
  const logoutButton = document.querySelector("#logout");
  if (currentUser) {
    loginButton.textContent = currentUser.display_name || currentUser.email;
    historyButton.classList.remove("hidden");
    logoutButton.classList.remove("hidden");
  } else {
    loginButton.textContent = "登录 / 注册";
    historyButton.classList.add("hidden");
    logoutButton.classList.add("hidden");
  }
  if (allRecommendations.length) renderPage();
}

async function loadSession() {
  try {
    const response = await fetch("/api/auth/me");
    const data = await response.json();
    currentUser = data.user || null;
  } catch {
    currentUser = null;
  }
  renderAuthState();
}

async function submitAuth(mode) {
  const message = document.querySelector("#auth-message");
  message.textContent = "";
  const body = {
    email: document.querySelector("#auth-email").value,
    password: document.querySelector("#auth-password").value,
    display_name: document.querySelector("#auth-name").value,
  };
  try {
    const response = await fetch(`/api/auth/${mode}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(await readErrorMessage(response));
    const data = await response.json();
    currentUser = data.user;
    closeModal("auth-modal");
    renderAuthState();
  } catch (error) {
    message.textContent = error.message;
  }
}

async function logout() {
  await fetch("/api/auth/logout", {method: "POST"});
  currentUser = null;
  renderAuthState();
}

function defaultFormulaName(item) {
  const formula = (item.components || [])
    .map(component => component.code)
    .join(" + ");
  const now = new Date().toLocaleString("zh-CN", {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"});
  return `${formula || "候选配方"} · ${now}`;
}

async function saveRecommendation(index) {
  if (!currentUser) {
    openModal("auth-modal");
    return;
  }
  const item = allRecommendations[index];
  if (!item) return;
  const name = window.prompt("给这个配方起个名字：", defaultFormulaName(item));
  if (!name) return;
  try {
    const response = await fetch("/api/history", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        name,
        recommendation: item,
        request_context: lastRequestBody || {},
      }),
    });
    if (!response.ok) throw new Error(await readErrorMessage(response));
    document.querySelector("#notice").textContent = `已保存到历史配方：${name}`;
    document.querySelector("#notice").classList.remove("hidden");
  } catch (error) {
    window.alert(`保存失败：${error.message}`);
  }
}

function renderHistory(items) {
  const list = document.querySelector("#history-list");
  if (!items.length) {
    list.innerHTML = `<div class="empty-history">还没有保存配方。先运行筛选，然后点击候选卡片里的“保存配方”。</div>`;
    return;
  }
  list.innerHTML = items.map(item => {
    const created = new Date(item.created_at).toLocaleString("zh-CN");
    const components = item.recommendation?.components || [];
    const ratios = components.map(component => `${component.code} ${component.ratio}%`).join(" · ");
    return `
      <article class="history-item">
        <div>
          <h3>${escapeHtml(item.name)}</h3>
          <p>${escapeHtml(item.formula)}</p>
          <p>${escapeHtml(ratios || "历史配方")}</p>
          <div class="history-meta">
            <span>综合评分 ${item.score ?? "-"}</span>
            <span>置信度 ${item.confidence ?? "-"}%</span>
            <span>${created}</span>
          </div>
        </div>
        <button class="small-action danger" data-delete-history="${item.id}" type="button">删除</button>
      </article>`;
  }).join("");
}

async function loadHistory() {
  if (!currentUser) {
    openModal("auth-modal");
    return;
  }
  openModal("history-modal");
  const list = document.querySelector("#history-list");
  list.innerHTML = "正在读取历史配方……";
  try {
    const response = await fetch("/api/history");
    if (!response.ok) throw new Error(await readErrorMessage(response));
    const data = await response.json();
    renderHistory(data.items || []);
  } catch (error) {
    list.innerHTML = `读取失败：${error.message}`;
  }
}

async function deleteHistoryItem(id) {
  if (!window.confirm("确定删除这条历史配方吗？")) return;
  const response = await fetch(`/api/history/${id}`, {method: "DELETE"});
  if (response.ok) loadHistory();
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><b>${value}</b></div>`;
}

function renderFeasibilityAdvice(advice) {
  const box = document.querySelector("#feasibility-advice");
  if (!advice || !Array.isArray(advice.suggestions) || !advice.suggestions.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  const rows = advice.suggestions.map(item => `
    <div class="advice-item">
      <div class="advice-label">${escapeHtml(item.label)}</div>
      <div class="advice-change">${escapeHtml(item.current)} → ${escapeHtml(item.suggested)}</div>
      <div class="advice-reason">${escapeHtml(item.reason)}</div>
      ${item.suggested_value === undefined ? "" : `<button class="apply-advice" type="button" data-advice-parameter="${escapeHtml(item.parameter)}" data-advice-value="${escapeHtml(item.suggested_value)}">采用</button>`}
    </div>
  `).join("");
  box.innerHTML = `
    <h3>${escapeHtml(advice.title || "当前目标组合过于严格")}</h3>
    <p>${escapeHtml(advice.message || "系统已返回最接近候选，并建议放宽以下条件。")}</p>
    <div class="advice-list">${rows}</div>
  `;
  box.classList.remove("hidden");
}

function applyAdvice(parameter, value) {
  const fieldMap = {
    min_flash_point_c: "flash",
    max_mixture_viscosity: "viscosity",
    score_threshold: "score-threshold",
  };
  const field = fieldMap[parameter];
  if (!field) return;
  document.querySelector(`#${field}`).value = value;
  saveSettings();
  runScreening();
}

function confidenceText(item) {
  const labels = {high: "较高", medium: "中等", low: "较低"};
  return labels[item.confidence_level] || item.confidence_label || "参考";
}

function renderScoreBreakdown(item) {
  const targets = Object.values(item.score_breakdown?.targets || {});
  if (!targets.length) return "";
  const rows = targets.map(target => `
    <div class="breakdown-row">
      <span>${escapeHtml(target.label)} <small>权重 ${target.weight_percent}%</small></span>
      <i><b style="width:${Math.max(0, Math.min(100, target.property_score))}%"></b></i>
      <strong>+${Number(target.contribution).toFixed(1)}</strong>
    </div>
  `).join("");
  const adjustment = Number(item.score_breakdown.adjustment || 0);
  return `
    <details class="score-explain">
      <summary>为什么是 ${item.score} 分 · 主要由${escapeHtml(item.score_breakdown.dominant_goal_label)}贡献</summary>
      <div class="breakdown-list">${rows}</div>
      ${Math.abs(adjustment) < 0.05 ? "" : `<p>物态、互补性与约束修正：${adjustment > 0 ? "+" : ""}${adjustment.toFixed(1)} 分</p>`}
      <p>置信度表示数据覆盖与模型一致性，不参与综合评分排序。</p>
    </details>`;
}

function renderCard(item, index) {
  const p = item.properties;
  const components = item.components || [
    {code: item.solvent_a, name: item.solvent_a_name, ratio: item.ratio_a},
    {code: item.solvent_b, name: item.solvent_b_name, ratio: item.ratio_b},
  ];
  const formula = components.map(component => escapeHtml(component.code)).join(" + ");
  const names = components.map(component => escapeHtml(component.name)).join(" / ");
  const ratioSegments = components
    .map(component => `<i style="width:${component.ratio}%"></i>`)
    .join("");
  const ratioLabels = components
    .map(component => `<span>${escapeHtml(component.code)} ${component.ratio}%</span>`)
    .join("");
  const reasons = item.reasons.map(x => `<span class="reason">${escapeHtml(x)}</span>`).join("");
  const violations = (item.constraint_violations || [])
    .map(x => `<span class="reason violation">${escapeHtml(x)}</span>`).join("");
  const compatibilityNotes = (item.compatibility_notes || [])
    .map(x => `<span class="reason compatibility">兼容性提醒：${escapeHtml(x)}</span>`).join("");
  const factorLabels = {
    domain_similarity: "训练域相似",
    ensemble_agreement: "模型一致",
    local_data_density: "局部数据",
    temperature_coverage: "温度覆盖",
    component_coverage: "溶剂覆盖",
    physics_model_agreement: "机理一致",
    mixture_conductivity_domain: "电导训练域",
    mixture_solubility_domain: "溶解训练域",
    mixture_model_domain_mean: "配方训练域",
    oedb_md_coverage: "OEDB 覆盖",
  };
  const confidenceFactors = Object.entries(item.confidence_factors || {})
    .filter(([key]) => !key.endsWith("target_count"))
    .map(([key, value]) => `<span class="confidence-factor">${escapeHtml(factorLabels[key] || key)} ${Math.round(Number(value) * 100)}%</span>`)
    .join("");
  const conductivity = item.predicted_conductivity === null ? p.conductivity_score : item.predicted_conductivity;
  const conductivityLabel = item.predicted_conductivity === null ? "离子传输评分" : "预测电导率";
  const viscosityLabel = p.oedb_viscosity_mpas === undefined ? "混合黏度" : "OEDB-MD 黏度";
  const viscosityValue = p.oedb_viscosity_mpas === undefined
    ? `${p.viscosity_mpas} mPa·s`
    : `${p.oedb_viscosity_mpas} mPa·s`;
  const oedbExtraMetrics = p.oedb_density_g_cm3 === undefined ? "" : [
    metric("OEDB-MD 密度", `${p.oedb_density_g_cm3} g/cm³`),
    p.oedb_cation_diffusivity_m2_s === undefined
      ? ""
      : metric("OEDB-MD Li⁺扩散", Number(p.oedb_cation_diffusivity_m2_s).toExponential(2)),
    metric("OEDB 覆盖", `${p.oedb_md_coverage}%`),
  ].join("");
  const solubilityValue = item.predicted_solubility_mole_fraction === null
    ? p.solubility_score
    : item.predicted_solubility_mole_fraction;
  const solubilityLabel = item.predicted_solubility_mole_fraction === null
    ? "溶解评分"
    : "预测溶解度 x";
  const saveButton = currentUser
    ? `<button class="small-action save-formula" data-save-index="${index}" type="button">保存配方</button>`
    : "";
  const evidenceTags = (item.evidence_tags || [])
    .map(tag => `<span class="evidence-tag">${escapeHtml(tag)}</span>`)
    .join("");
  const confidenceLevel = item.confidence_level || "low";
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
          <div class="score-ring" style="--score:${item.score}%"><b>${item.score}</b><span>综合分</span></div>
          <div class="metrics">
            <div class="metric confidence-metric ${confidenceLevel}"><span>置信度 · ${confidenceText(item)}</span><b>${item.confidence}%</b></div>
            ${metric(conductivityLabel, conductivity)}
            ${metric(viscosityLabel, viscosityValue)}
            ${metric(solubilityLabel, solubilityValue)}
            ${metric("稳定评分", p.stability_score)}
            ${metric("估算闪点", `${p.flash_point_c} °C`)}
            ${oedbExtraMetrics}
          </div>
        </div>
      </div>
      <div class="card-detail">
        <div class="card-actions"><span class="basis">${escapeHtml(item.basis)}</span>${evidenceTags}${saveButton}</div>
        <div class="reasons">${confidenceFactors}${reasons}${compatibilityNotes}${violations}</div>
      </div>
      ${renderScoreBreakdown(item)}
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

function renderResultSummary(summary, runtime, searchSpace) {
  const box = document.querySelector("#result-summary");
  if (!summary || !summary.count) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  const elapsed = runtime?.elapsed_ms === undefined ? "-" : `${(runtime.elapsed_ms / 1000).toFixed(1)} 秒`;
  const refined = searchSpace?.model_refined_formulations || searchSpace?.evaluated_formulations || 0;
  box.innerHTML = `
    <div><span>最高综合分</span><b>${summary.top_score}</b></div>
    <div><span>中位置信度</span><b>${summary.median_confidence}%</b></div>
    <div><span>严格满足约束</span><b>${summary.strict_count} / ${summary.count}</b></div>
    <div><span>含 OEDB 模拟</span><b>${summary.oedb_count}</b></div>
    <div><span>规则预先排除</span><b>${(summary.compatibility_blocked_count || 0).toLocaleString()}</b></div>
    <div><span>${runtime?.cache_hit ? "缓存命中" : "模型精排"}</span><b>${runtime?.cache_hit ? elapsed : `${refined.toLocaleString()} 个`}</b></div>
  `;
  box.classList.remove("hidden");
}

function startLoadingClock() {
  const stage = document.querySelector("#loading-stage");
  const time = document.querySelector("#loading-time");
  const started = performance.now();
  clearInterval(loadingTimer);
  loadingTimer = setInterval(() => {
    const seconds = Math.floor((performance.now() - started) / 1000);
    time.textContent = `已用时 ${seconds} 秒`;
    if (seconds >= 12) stage.textContent = "正在进行模型精排，三元体系需要更久…";
    else if (seconds >= 4) stage.textContent = "正在计算性质并应用约束…";
  }, 500);
}

function stopLoadingClock() {
  clearInterval(loadingTimer);
  loadingTimer = null;
}

function validateScreeningBody(body) {
  if (!body.salt.trim()) return "请先输入目标锂盐。";
  if (!Number.isFinite(body.temperature_c) || !Number.isFinite(body.concentration)) return "请检查温度和盐浓度。";
  if (body.concentration <= 0) return "盐浓度必须大于 0。";
  const total = Object.values(body.weights).reduce((sum, value) => sum + value, 0);
  if (Math.abs(total - 1) > 0.001) return "五项目标权重的总和必须为 100%。";
  return "";
}

async function runScreening() {
  const button = document.querySelector("#run");
  const loading = document.querySelector("#loading");
  const empty = document.querySelector("#empty");
  const cards = document.querySelector("#cards");
  const notice = document.querySelector("#notice");
  const pager = document.querySelector("#pager");
  const adviceBox = document.querySelector("#feasibility-advice");
  const resultSummary = document.querySelector("#result-summary");
  if (activeRequestController) activeRequestController.abort();
  const requestController = new AbortController();
  activeRequestController = requestController;
  button.disabled = true;
  button.innerHTML = "正在筛选 <span>···</span>";
  empty.classList.add("hidden");
  cards.innerHTML = "";
  allRecommendations = [];
  currentPage = 1;
  pager.classList.add("hidden");
  notice.classList.add("hidden");
  adviceBox.classList.add("hidden");
  adviceBox.innerHTML = "";
  resultSummary.classList.add("hidden");
  loading.classList.remove("hidden");
  document.querySelector("#loading-stage").textContent = "正在生成候选组合…";
  document.querySelector("#loading-time").textContent = "已用时 0 秒";
  startLoadingClock();

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
  const validationError = validateScreeningBody(body);
  if (validationError) {
    notice.textContent = validationError;
    notice.classList.remove("hidden");
    loading.classList.add("hidden");
    stopLoadingClock();
    button.disabled = false;
    button.innerHTML = "开始筛选 <span>→</span>";
    activeRequestController = null;
    return;
  }
  lastRequestBody = body;
  saveSettings();

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
      signal: requestController.signal,
    });
    if (!response.ok) throw new Error(await readErrorMessage(response));
    const data = await response.json();
    notice.textContent = data.warning || "";
    notice.classList.toggle("hidden", !data.warning);
    renderFeasibilityAdvice(data.feasibility_advice);
    document.querySelector("#search-stat").textContent =
      `${data.search_space.solvents} 种溶剂 · 全空间 ${data.search_space.evaluated_formulations.toLocaleString()} 个 · 返回 ${data.recommendations.length} 个`;
    allRecommendations = data.recommendations;
    renderResultSummary(data.result_summary, data.runtime, data.search_space);
    renderPage();
    if (!data.recommendations.length) {
      empty.querySelector("h3").textContent = "当前约束下没有可行结果";
      empty.querySelector("p").textContent = "可适当降低最低闪点或提高最高黏度限制。";
      empty.classList.remove("hidden");
    }
  } catch (error) {
    if (error.name === "AbortError") return;
    notice.textContent = `运行失败：${error.message}`;
    notice.classList.remove("hidden");
    empty.classList.remove("hidden");
  } finally {
    if (activeRequestController === requestController) {
      loading.classList.add("hidden");
      stopLoadingClock();
      button.disabled = false;
      button.innerHTML = "开始筛选 <span>→</span>";
      activeRequestController = null;
    }
  }
}

document.querySelector("#run").addEventListener("click", runScreening);
document.querySelector("#auth-open").addEventListener("click", () => openModal("auth-modal"));
document.querySelector("#history-open").addEventListener("click", loadHistory);
document.querySelector("#logout").addEventListener("click", logout);
document.querySelector("#login").addEventListener("click", () => submitAuth("login"));
document.querySelector("#register").addEventListener("click", () => submitAuth("register"));
document.querySelectorAll("[data-close]").forEach(button => {
  button.addEventListener("click", () => closeModal(button.dataset.close));
});
document.querySelector("#cards").addEventListener("click", event => {
  const button = event.target.closest("[data-save-index]");
  if (button) saveRecommendation(Number(button.dataset.saveIndex));
});
document.querySelector("#feasibility-advice").addEventListener("click", event => {
  const button = event.target.closest("[data-advice-parameter]");
  if (button) applyAdvice(button.dataset.adviceParameter, button.dataset.adviceValue);
});
document.querySelector("#history-list").addEventListener("click", event => {
  const button = event.target.closest("[data-delete-history]");
  if (button) deleteHistoryItem(Number(button.dataset.deleteHistory));
});
document.querySelector("#prev-page").addEventListener("click", () => {
  currentPage -= 1;
  renderPage();
  document.querySelector(".results-head").scrollIntoView({behavior: "smooth", block: "start"});
});
document.querySelector("#next-page").addEventListener("click", () => {
  currentPage += 1;
  renderPage();
  document.querySelector(".results-head").scrollIntoView({behavior: "smooth", block: "start"});
});
document.querySelector(".controls").addEventListener("change", saveSettings);
loadModelInfo();
loadSession();
