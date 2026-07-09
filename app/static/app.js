let allRecommendations = [];
let currentPage = 1;
const pageSize = 10;
let currentUser = null;
let lastRequestBody = null;

const scenarioWeights = {
  lithium_metal: {
    solubility: 0.34,
    conductivity: 0.24,
    stability: 0.22,
    safety: 0.12,
    low_temperature: 0.08,
  },
  high_voltage: {
    solubility: 0.24,
    conductivity: 0.24,
    stability: 0.30,
    safety: 0.14,
    low_temperature: 0.08,
  },
  balanced: {
    solubility: 0.28,
    conductivity: 0.26,
    stability: 0.18,
    safety: 0.16,
    low_temperature: 0.12,
  },
};

function currentScenarioWeights() {
  return scenarioWeights[document.querySelector("#application").value] || scenarioWeights.balanced;
}

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
    const solventText = solventCount ? ` · 候选溶剂 ${solventCount} 种` : "";
    if (mixture?.available) {
      pill.textContent = oedb?.available
        ? `配方模型在线 · 公开实验 ${mixtureRows.toLocaleString()} 条 · OEDB-MD ${oedbRows.toLocaleString()} 条 · LiNO₃ 二元 ${lino3BinaryRows} 条${solventText}`
        : `配方模型在线 · 公开实验 ${mixtureRows.toLocaleString()} 条 · LiNO₃ 二元 ${lino3BinaryRows} 条${solventText}`;
    } else if (info.available) {
      pill.textContent = `模型在线 · 电导率 ${info.metrics.train_rows.toLocaleString()} 条 · LiNO₃ 溶解度 ${lino3Rows} 条`;
    } else if (lino3Rows > 0) {
      pill.textContent = `轻量模型在线 · LiNO₃ 溶解度 ${lino3Rows} 条`;
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
    </div>
  `).join("");
  box.innerHTML = `
    <h3>${escapeHtml(advice.title || "当前目标组合过于严格")}</h3>
    <p>${escapeHtml(advice.message || "系统已返回最接近候选，并建议放宽以下条件。")}</p>
    <div class="advice-list">${rows}</div>
  `;
  box.classList.remove("hidden");
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
  const conductivity = item.predicted_conductivity === null
    ? `${p.conductivity_estimate_ms_cm} mS/cm`
    : `${item.predicted_conductivity} mS/cm`;
  const conductivityLabel = item.predicted_conductivity === null ? "估算电导率" : "预测电导率";
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
            ${metric(viscosityLabel, viscosityValue)}
            ${metric(solubilityLabel, solubilityValue)}
            ${metric("稳定评分", p.stability_score)}
            ${metric("估算闪点", `${p.flash_point_c} °C`)}
            ${oedbExtraMetrics}
          </div>
        </div>
      </div>
      <div class="card-detail">
        <div class="card-actions"><span class="basis">${item.basis}</span>${saveButton}</div>
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
  const adviceBox = document.querySelector("#feasibility-advice");
  button.disabled = true;
  empty.classList.add("hidden");
  cards.innerHTML = "";
  allRecommendations = [];
  currentPage = 1;
  pager.classList.add("hidden");
  notice.classList.add("hidden");
  adviceBox.classList.add("hidden");
  adviceBox.innerHTML = "";
  loading.classList.remove("hidden");

  const body = {
    salt: document.querySelector("#salt").value,
    temperature_c: Number(document.querySelector("#temperature").value),
    concentration: Number(document.querySelector("#concentration").value),
    concentration_unit: "mol/kg",
    application: document.querySelector("#application").value,
    min_solubility_score: Number(document.querySelector("#solubility-target").value),
    min_conductivity_ms_cm: Number(document.querySelector("#conductivity-target").value),
    min_flash_point_c: Number(document.querySelector("#flash").value),
    max_mixture_viscosity: Number(document.querySelector("#viscosity").value),
    min_stability_score: Number(document.querySelector("#stability-target").value),
    min_safety_score: Number(document.querySelector("#safety-target").value),
    min_low_temperature_score: Number(document.querySelector("#low-temperature-target").value),
    exclude_high_hazard: document.querySelector("#hazard").checked,
    top_k: 10,
    score_threshold: Number(document.querySelector("#score-threshold").value),
    max_results: 120,
    max_components: Number(document.querySelector("#max-components").value),
    return_all_above_threshold: true,
    allow_relaxed_fallback: true,
    weights: currentScenarioWeights(),
  };
  lastRequestBody = body;

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
    renderFeasibilityAdvice(data.feasibility_advice);
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
document.querySelector("#history-list").addEventListener("click", event => {
  const button = event.target.closest("[data-delete-history]");
  if (button) deleteHistoryItem(Number(button.dataset.deleteHistory));
});
document.querySelector("#prev-page").addEventListener("click", () => {
  currentPage -= 1;
  renderPage();
});
document.querySelector("#next-page").addEventListener("click", () => {
  currentPage += 1;
  renderPage();
});
loadModelInfo();
loadSession();
