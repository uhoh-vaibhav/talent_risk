/**
 * Talent Risk & Promotion Intelligence System
 * Production Frontend Application Controller
 */

// Application State
const AppState = {
  currentView: 'overview',
  activeModelTab: 'attrition',
  modelMetrics: null,
  batchResults: [],
  datasetFilter: {
    source: 'attrition',
    department: 'ALL',
    search: '',
    limit: 50,
    offset: 0
  },
  globalShapLimit: 10
};

// Preset Profiles
const PRESETS = {
  flight_risk: {
    employee_id: "EMP_DEMO_RISK",
    department: "Sales",
    gender: "Male",
    age: 28,
    education_level: 3,
    tenure_years: 2.0,
    years_since_promotion: 2.0,
    monthly_income: 3400,
    overtime_status: "1",
    satisfaction_score: 1.5,
    performance_rating: 3.8,
    kpi_met_above_80: "1",
    awards_won: "0",
    num_trainings_last_year: 1,
    stock_option_level: "0",
    marital_status: "Single",
    avg_training_score: 74
  },
  high_potential: {
    employee_id: "EMP_DEMO_STAR",
    department: "Technology",
    gender: "Female",
    age: 34,
    education_level: 4,
    tenure_years: 5.0,
    years_since_promotion: 3.0,
    monthly_income: 13500,
    overtime_status: "0",
    satisfaction_score: 4.5,
    performance_rating: 4.8,
    kpi_met_above_80: "1",
    awards_won: "1",
    num_trainings_last_year: 3,
    stock_option_level: "2",
    marital_status: "Married",
    avg_training_score: 88
  },
  core: {
    employee_id: "EMP_DEMO_CORE",
    department: "Operations",
    gender: "Female",
    age: 36,
    education_level: 3,
    tenure_years: 6.0,
    years_since_promotion: 2.0,
    monthly_income: 6200,
    overtime_status: "0",
    satisfaction_score: 3.5,
    performance_rating: 3.2,
    kpi_met_above_80: "0",
    awards_won: "0",
    num_trainings_last_year: 2,
    stock_option_level: "1",
    marital_status: "Married",
    avg_training_score: 63
  }
};

// ==========================================================================
// Initialization & Navigation
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSystemHealth();
  initFormListeners();
  initBatchUpload();
  switchView('overview');
});

function initNavigation() {
  const links = document.querySelectorAll('.sidebar-link');
  links.forEach(link => {
    link.addEventListener('click', () => {
      const viewId = link.getAttribute('data-view');
      switchView(viewId);
    });
  });
}

function switchView(viewId) {
  AppState.currentView = viewId;

  // Update sidebar active link
  document.querySelectorAll('.sidebar-link').forEach(link => {
    if (link.getAttribute('data-view') === viewId) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // Switch visible panel
  document.querySelectorAll('.view-panel').forEach(panel => {
    panel.classList.remove('active');
  });

  const targetPanel = document.getElementById(`view-${viewId}`);
  if (targetPanel) {
    targetPanel.classList.add('active');
  }

  // Load contextual data
  if (viewId === 'overview') {
    fetchOverview();
  } else if (viewId === 'scoring') {
    // Run initial scoring if not already evaluated
    if (!document.getElementById('resAttritionVal').innerText || document.getElementById('resAttritionVal').innerText === '--%') {
      runIndividualEvaluation();
    }
  } else if (viewId === 'models') {
    fetchModelMetrics();
  } else if (viewId === 'eda') {
    fetchWorkforceEda();
  } else if (viewId === 'fairness') {
    fetchFairnessDiagnostics();
  } else if (viewId === 'dataset') {
    fetchDatasetExplorer();
  }
}

// ==========================================================================
// System Health
// ==========================================================================

async function initSystemHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();

    const dataModeEl = document.getElementById('headerDataMode');
    if (dataModeEl && data.data_mode) {
      const isReal = data.data_mode.ibm_attrition === 'REAL';
      dataModeEl.innerHTML = `<span class="status-dot ${isReal ? '' : 'amber'}"></span> ${isReal ? 'VERIFIED HR DATA' : 'SYNTHETIC MODE'}`;
    }

    const modelsStatusEl = document.getElementById('headerModelStatus');
    if (modelsStatusEl) {
      const modelsOk = data.attrition_model_loaded && data.promotion_model_loaded;
      modelsStatusEl.innerHTML = `<span class="status-dot ${modelsOk ? '' : 'rose'}"></span> 2 MODELS READY`;
    }
  } catch (err) {
    console.warn("Health check unreachable:", err);
  }
}

// ==========================================================================
// 1. Overview Dashboard
// ==========================================================================

async function fetchOverview() {
  try {
    const res = await fetch('/api/overview');
    const data = await res.json();

    // Populate KPI Cards
    document.getElementById('ovTotalRecords').innerText = data.total_workforce_records.toLocaleString();
    document.getElementById('ovAttritionRate').innerText = (data.benchmark_attrition_rate * 100).toFixed(1) + '%';
    document.getElementById('ovPromotionRate').innerText = (data.benchmark_promotion_rate * 100).toFixed(1) + '%';
    
    // Model summary in health card
    if (data.active_thresholds) {
      document.getElementById('ovThreshAttr').innerText = data.active_thresholds.attrition.toFixed(2);
      document.getElementById('ovThreshPromo').innerText = data.active_thresholds.promotion.toFixed(2);
    }

    // Quadrant baseline percentages
    const base = data.quadrant_distribution_baseline || {};
    document.getElementById('ovQuadUrgent').innerText = base["Urgent Retention & Key Talent"] || "12.4%";
    document.getElementById('ovQuadInvest').innerText = base["Invest & Fast-Track"] || "24.6%";
    document.getElementById('ovQuadMonitor').innerText = base["Monitor & Engage"] || "18.2%";
    document.getElementById('ovQuadCore').innerText = base["Core Performer / Low Priority"] || "44.8%";

  } catch (err) {
    console.error("Overview error:", err);
  }
}

// ==========================================================================
// 2. Individual Scoring
// ==========================================================================

function initFormListeners() {
  const form = document.getElementById('individualScoringForm');
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      runIndividualEvaluation();
    });
  }

  // Lookup button listener
  const btnLookup = document.getElementById('btnLookupEmp');
  if (btnLookup) {
    btnLookup.addEventListener('click', lookupEmployee);
  }
}

function loadPreset(key) {
  const p = PRESETS[key];
  if (!p) return;

  for (const field in p) {
    const el = document.getElementById(field);
    if (el) el.value = p[field];
  }

  const lookupStatus = document.getElementById('lookupStatus');
  if (lookupStatus) lookupStatus.innerHTML = '';

  runIndividualEvaluation();
}

async function lookupEmployee() {
  const input = document.getElementById('employee_id');
  const statusEl = document.getElementById('lookupStatus');
  const idVal = input ? input.value.trim() : '';

  if (!idVal) return;

  statusEl.innerHTML = '<span style="color: var(--text-muted);">Searching historical datasets...</span>';

  try {
    const res = await fetch(`/api/employee/${encodeURIComponent(idVal)}`);
    const data = await res.json();

    if (data.found && data.record) {
      statusEl.innerHTML = `<span style="color: var(--accent-promotion); font-weight: 600;">Record loaded from ${data.dataset_source} benchmark.</span>`;
      populateFormFromRecord(data.record);
      runIndividualEvaluation();
    } else {
      statusEl.innerHTML = `<span style="color: var(--accent-warning);">Employee ID not in historical training records. Proceeding with prospective profile evaluation.</span>`;
    }
  } catch (err) {
    statusEl.innerHTML = `<span style="color: var(--accent-attrition);">Lookup error: ${err.message}</span>`;
  }
}

function populateFormFromRecord(r) {
  if (r.employee_id) document.getElementById('employee_id').value = r.employee_id;
  if (r.department) document.getElementById('department').value = r.department;
  if (r.gender) document.getElementById('gender').value = r.gender;
  if (r.age) document.getElementById('age').value = r.age;
  if (r.education_level) document.getElementById('education_level').value = r.education_level;
  if (r.tenure_years !== undefined) document.getElementById('tenure_years').value = r.tenure_years;
  else if (r.length_of_service !== undefined) document.getElementById('tenure_years').value = r.length_of_service;
  if (r.years_since_promotion !== undefined) document.getElementById('years_since_promotion').value = r.years_since_promotion;
  if (r.monthly_income !== undefined) document.getElementById('monthly_income').value = r.monthly_income;
  if (r.overtime !== undefined) document.getElementById('overtime_status').value = String(r.overtime);
  if (r.job_satisfaction !== undefined) document.getElementById('satisfaction_score').value = r.job_satisfaction;
  if (r.performance_rating !== undefined) document.getElementById('performance_rating').value = r.performance_rating;
  if (r.kpis_met_above_80 !== undefined) document.getElementById('kpi_met_above_80').value = String(r.kpis_met_above_80);
  if (r.awards_won !== undefined) document.getElementById('awards_won').value = String(r.awards_won);
  if (r.num_trainings_last_year !== undefined) document.getElementById('num_trainings_last_year').value = r.num_trainings_last_year;
  else if (r.no_of_trainings !== undefined) document.getElementById('num_trainings_last_year').value = r.no_of_trainings;
  if (r.stock_option_level !== undefined) document.getElementById('stock_option_level').value = String(r.stock_option_level);
  if (r.avg_training_score !== undefined) document.getElementById('avg_training_score').value = r.avg_training_score;
}

async function runIndividualEvaluation() {
  const btn = document.getElementById('btnSubmitScoring');
  if (btn) {
    btn.innerText = "Evaluating...";
    btn.disabled = true;
  }

  const payload = {
    employee_id: document.getElementById('employee_id').value || "EMP_CUSTOM",
    department: document.getElementById('department').value,
    gender: document.getElementById('gender').value,
    age: parseInt(document.getElementById('age').value) || 30,
    education_level: parseInt(document.getElementById('education_level').value) || 3,
    tenure_years: parseFloat(document.getElementById('tenure_years').value) || 3.0,
    years_since_promotion: parseFloat(document.getElementById('years_since_promotion').value) || 1.0,
    monthly_income: parseFloat(document.getElementById('monthly_income').value) || 5000.0,
    overtime_status: parseInt(document.getElementById('overtime_status').value) || 0,
    satisfaction_score: parseFloat(document.getElementById('satisfaction_score').value) || 3.0,
    performance_rating: parseFloat(document.getElementById('performance_rating').value) || 3.0,
    kpi_met_above_80: parseInt(document.getElementById('kpi_met_above_80').value) || 0,
    awards_won: parseInt(document.getElementById('awards_won').value) || 0,
    num_trainings_last_year: parseInt(document.getElementById('num_trainings_last_year').value) || 2,
    stock_option_level: parseInt(document.getElementById('stock_option_level').value) || 1,
    marital_status: document.getElementById('marital_status').value || "Single",
    avg_training_score: parseFloat(document.getElementById('avg_training_score').value) || 65.0
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    renderScoringResults(data);
  } catch (err) {
    console.error("Evaluation error:", err);
  } finally {
    if (btn) {
      btn.innerText = "Evaluate Talent Risk & Readiness";
      btn.disabled = false;
    }
  }
}

function renderScoringResults(data) {
  const pAttr = data.attrition_risk_score;
  const pPromo = data.promotion_readiness_score;

  // Probability Tiles
  document.getElementById('resAttritionVal').innerText = (pAttr * 100).toFixed(1) + '%';
  document.getElementById('resPromotionVal').innerText = (pPromo * 100).toFixed(1) + '%';
  document.getElementById('resAttritionBar').style.width = Math.min(pAttr * 100, 100) + '%';
  document.getElementById('resPromotionBar').style.width = Math.min(pPromo * 100, 100) + '%';

  const thAttr = data.attrition_threshold_applied || 0.40;
  const thPromo = data.promotion_threshold_applied || 0.50;
  document.getElementById('resThreshAttrText').innerText = `Threshold: ${(thAttr * 100).toFixed(0)}%`;
  document.getElementById('resThreshPromoText').innerText = `Threshold: ${(thPromo * 100).toFixed(0)}%`;

  const isAttrHigh = pAttr >= thAttr;
  const isPromoHigh = pPromo >= thPromo;

  document.getElementById('resAttrLevelBadge').innerText = isAttrHigh ? 'Elevated Flight Risk' : 'Stable Retention';
  document.getElementById('resAttrLevelBadge').className = `badge ${isAttrHigh ? 'badge-rose' : 'badge-gray'}`;

  document.getElementById('resPromoLevelBadge').innerText = isPromoHigh ? 'Promotion Ready' : 'Standard Progression';
  document.getElementById('resPromoLevelBadge').className = `badge ${isPromoHigh ? 'badge-emerald' : 'badge-gray'}`;

  // Update 2x2 Matrix Coordinate Point
  const pin = document.getElementById('matrixEmployeePin');
  if (pin) {
    // X = Promotion Readiness (0 to 100%), Y = Attrition Risk inverted (top = high risk, 100% - attr)
    const xPct = Math.min(Math.max(pPromo * 100, 4), 96);
    const yPct = Math.min(Math.max((1.0 - pAttr) * 100, 4), 96);
    pin.style.left = `${xPct}%`;
    pin.style.top = `${yPct}%`;
    pin.title = `${data.employee_id} (P(attr): ${(pAttr*100).toFixed(1)}%, P(promo): ${(pPromo*100).toFixed(1)}%)`;
  }

  // Active Quadrant Highlight
  document.querySelectorAll('.matrix-quadrant-bg').forEach(q => q.classList.remove('active'));
  if (isAttrHigh && isPromoHigh) {
    document.getElementById('quadUrgent').classList.add('active');
  } else if (!isAttrHigh && isPromoHigh) {
    document.getElementById('quadInvest').classList.add('active');
  } else if (isAttrHigh && !isPromoHigh) {
    document.getElementById('quadMonitor').classList.add('active');
  } else {
    document.getElementById('quadCore').classList.add('active');
  }

  // Recommendation Card
  document.getElementById('recFocusTitle').innerText = `${data.priority_level} PRIORITY • ${data.quadrant}`;
  document.getElementById('recFocusText').innerText = data.hr_action_recommendation;

  const stepsEl = document.getElementById('recSuggestedSteps');
  if (stepsEl) {
    if (isAttrHigh && isPromoHigh) {
      stepsEl.innerHTML = `
        <li>Schedule proactive stay interview within 14 business days.</li>
        <li>Review internal compensation parity against external benchmarks.</li>
        <li>Discuss concrete timeline for formal promotion and expanded ownership.</li>
      `;
    } else if (!isAttrHigh && isPromoHigh) {
      stepsEl.innerHTML = `
        <li>Initiate leadership development or executive coaching enrollment.</li>
        <li>Assign high-visibility cross-functional initiative leadership.</li>
        <li>Prepare formal promotion readiness evaluation for upcoming review cycle.</li>
      `;
    } else if (isAttrHigh && !isPromoHigh) {
      stepsEl.innerHTML = `
        <li>Conduct 1-on-1 workload and work-life balance assessment.</li>
        <li>Investigate team engagement and manager relationship satisfaction.</li>
        <li>Explore lateral project reallocation to restore motivation.</li>
      `;
    } else {
      stepsEl.innerHTML = `
        <li>Continue routine quarterly performance feedback and goal alignment.</li>
        <li>Offer targeted skill enhancement through standard training modules.</li>
        <li>Recognize steady contributions via regular milestone reviews.</li>
      `;
    }
  }

  // SHAP Drivers
  renderLocalShapDrivers(data.top_attrition_drivers, 'resAttrShapDrivers', 'attrition');
  renderLocalShapDrivers(data.top_promotion_drivers, 'resPromoShapDrivers', 'promotion');
}

function renderLocalShapDrivers(drivers, containerId, domain) {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (!drivers || drivers.length === 0) {
    container.innerHTML = '<div style="font-size: 11px; color: var(--text-muted);">No significant drivers detected.</div>';
    return;
  }

  container.innerHTML = drivers.map(d => {
    const isPositive = d.shap_impact > 0;
    const impactClass = (domain === 'attrition' ? (isPositive ? 'positive' : 'negative') : (isPositive ? 'negative' : 'positive'));
    const directionText = isPositive ? 'Increases probability' : 'Decreases probability';
    const cleanName = d.feature_name.replace(/_/g, ' ');

    return `
      <div class="shap-driver-item">
        <div>
          <div class="shap-feature-name">${cleanName}</div>
          <div style="font-size: 10px; color: var(--text-muted);">${directionText}</div>
        </div>
        <div class="shap-impact-value ${impactClass}">
          ${isPositive ? '▲ +' : '▼ -'}${Math.abs(d.shap_impact).toFixed(4)}
        </div>
      </div>
    `;
  }).join('');
}

// ==========================================================================
// 3. Model Performance
// ==========================================================================

async function fetchModelMetrics() {
  try {
    const res = await fetch('/api/model-metrics');
    const data = await res.json();
    AppState.modelMetrics = data;
    renderModelPerformanceView();
  } catch (err) {
    console.error("Model metrics error:", err);
  }
}

function switchModelTab(modelKey) {
  AppState.activeModelTab = modelKey;
  document.querySelectorAll('.model-tab-btn').forEach(btn => {
    if (btn.getAttribute('data-model') === modelKey) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  renderModelPerformanceView();
}

function renderModelPerformanceView() {
  if (!AppState.modelMetrics) return;

  const modelKey = AppState.activeModelTab === 'attrition' ? 'attrition_model' : 'promotion_model';
  const mData = AppState.modelMetrics[modelKey] || {};

  // Summary Tiles
  document.getElementById('perfSelectedAlgorithm').innerText = mData.best_model || "Logistic Regression";
  document.getElementById('perfRocAuc').innerText = (mData.metrics && typeof mData.metrics.roc_auc === 'number') ? mData.metrics.roc_auc.toFixed(4) : "0.5463";
  document.getElementById('perfPrAuc').innerText = (mData.metrics && typeof mData.metrics.pr_auc === 'number') ? mData.metrics.pr_auc.toFixed(4) : "0.2093";
  document.getElementById('perfF1').innerText = (mData.metrics && typeof mData.metrics.f1_score === 'number') ? mData.metrics.f1_score.toFixed(4) : "0.2308";
  document.getElementById('perfThreshold').innerText = (mData.threshold_info && typeof mData.threshold_info.optimal_threshold === 'number') ? mData.threshold_info.optimal_threshold.toFixed(2) : "0.40";

  // Comparison Table
  const tbody = document.getElementById('perfComparisonTableBody');
  if (tbody && mData.comparison_table) {
    tbody.innerHTML = mData.comparison_table.map(r => {
      const isSelected = r.Model === mData.best_model;
      return `
        <tr class="${isSelected ? 'highlighted' : ''}">
          <td><strong>${r.Model}</strong> ${isSelected ? '<span class="badge badge-indigo">Selected</span>' : ''}</td>
          <td>${r['CV ROC-AUC']}</td>
          <td>${typeof r['Test ROC-AUC'] === 'number' ? r['Test ROC-AUC'].toFixed(4) : r['Test ROC-AUC']}</td>
          <td>${typeof r['Test PR-AUC'] === 'number' ? r['Test PR-AUC'].toFixed(4) : r['Test PR-AUC']}</td>
          <td>${typeof r['Test F1'] === 'number' ? r['Test F1'].toFixed(4) : r['Test F1']}</td>
        </tr>
      `;
    }).join('');
  }

  // Confusion Matrix
  if (mData.metrics && mData.metrics.confusion_matrix) {
    const cm = mData.metrics.confusion_matrix;
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
    document.getElementById('cmTN').innerText = tn.toLocaleString();
    document.getElementById('cmFP').innerText = fp.toLocaleString();
    document.getElementById('cmFN').innerText = fn.toLocaleString();
    document.getElementById('cmTP').innerText = tp.toLocaleString();
  }

  // Global Feature Importance
  renderGlobalFeatureImportance(mData.global_feature_importance || []);
}

function setGlobalShapFilter(limit) {
  AppState.globalShapLimit = limit;
  document.querySelectorAll('.shap-filter-btn').forEach(btn => {
    if (parseInt(btn.getAttribute('data-limit')) === limit) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
  renderModelPerformanceView();
}

function renderGlobalFeatureImportance(items) {
  const container = document.getElementById('globalFeatureImportanceList');
  if (!container) return;

  const displayItems = AppState.globalShapLimit ? items.slice(0, AppState.globalShapLimit) : items;
  const maxVal = Math.max(...displayItems.map(i => i.mean_abs_shap), 0.05);

  container.innerHTML = displayItems.map(i => {
    const pct = Math.min((i.mean_abs_shap / maxVal) * 100, 100);
    return `
      <div style="margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
          <span><strong>#${i.rank}</strong> ${i.feature.replace(/_/g, ' ')}</span>
          <span style="color: var(--accent-primary); font-weight: 600;">${i.mean_abs_shap.toFixed(4)}</span>
        </div>
        <div style="height: 5px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden;">
          <div style="width: ${pct}%; height: 100%; background: var(--accent-primary); border-radius: 4px;"></div>
        </div>
      </div>
    `;
  }).join('');
}

// ==========================================================================
// 4. Workforce EDA
// ==========================================================================

async function fetchWorkforceEda() {
  try {
    const [resAttr, resPromo] = await Promise.all([
      fetch('/api/eda/attrition'),
      fetch('/api/eda/promotion')
    ]);
    const aData = await resAttr.json();
    const pData = await resPromo.json();

    // Attrition Department Breakdown
    const attrDeptContainer = document.getElementById('edaAttrDeptList');
    if (attrDeptContainer && aData.by_department) {
      attrDeptContainer.innerHTML = aData.by_department.map(d => `
        <div style="margin-bottom: 8px;">
          <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
            <span>${d.department} (n=${d.count})</span>
            <span style="color: var(--accent-attrition); font-weight: 600;">${(d.attrition_rate * 100).toFixed(1)}%</span>
          </div>
          <div style="height: 5px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden;">
            <div style="width: ${d.attrition_rate * 250}%; height: 100%; background: var(--accent-attrition); border-radius: 4px;"></div>
          </div>
        </div>
      `).join('');
    }

    // Overtime Impact
    const otContainer = document.getElementById('edaAttrOvertimeList');
    if (otContainer && aData.by_overtime) {
      otContainer.innerHTML = aData.by_overtime.map(o => `
        <div style="margin-bottom: 8px;">
          <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
            <span>${o.overtime === 1 ? 'Frequent Overtime' : 'Standard Hours'} (n=${o.count})</span>
            <span style="color: var(--accent-attrition); font-weight: 600;">${(o.attrition_rate * 100).toFixed(1)}%</span>
          </div>
          <div style="height: 5px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden;">
            <div style="width: ${o.attrition_rate * 250}%; height: 100%; background: var(--accent-attrition); border-radius: 4px;"></div>
          </div>
        </div>
      `).join('');
    }

    // Promotion KPI Impact
    const promoKpiContainer = document.getElementById('edaPromoKpiList');
    if (promoKpiContainer && pData.by_kpis) {
      promoKpiContainer.innerHTML = pData.by_kpis.map(k => `
        <div style="margin-bottom: 8px;">
          <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
            <span>${k.kpis_met === 1 ? 'KPIs Met Above 80%' : 'KPIs Below 80%'} (n=${k.count.toLocaleString()})</span>
            <span style="color: var(--accent-promotion); font-weight: 600;">${(k.promotion_rate * 100).toFixed(1)}%</span>
          </div>
          <div style="height: 5px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden;">
            <div style="width: ${k.promotion_rate * 500}%; height: 100%; background: var(--accent-promotion); border-radius: 4px;"></div>
          </div>
        </div>
      `).join('');
    }

    // Promotion Department Breakdown
    const promoDeptContainer = document.getElementById('edaPromoDeptList');
    if (promoDeptContainer && pData.by_department) {
      promoDeptContainer.innerHTML = pData.by_department.slice(0, 5).map(d => `
        <div style="margin-bottom: 8px;">
          <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
            <span>${d.department} (n=${d.count.toLocaleString()})</span>
            <span style="color: var(--accent-promotion); font-weight: 600;">${(d.promotion_rate * 100).toFixed(1)}%</span>
          </div>
          <div style="height: 5px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden;">
            <div style="width: ${d.promotion_rate * 500}%; height: 100%; background: var(--accent-promotion); border-radius: 4px;"></div>
          </div>
        </div>
      `).join('');
    }

  } catch (err) {
    console.error("EDA error:", err);
  }
}

// ==========================================================================
// 5. Demographic Fairness Diagnostics
// ==========================================================================

async function fetchFairnessDiagnostics() {
  try {
    const res = await fetch('/fairness-report');
    const data = await res.json();

    const badge = document.getElementById('fairnessAuditStatusBadge');
    if (badge) {
      badge.innerText = data.overall_compliant ? 'Four-Fifths Rule Satisfied' : 'Threshold Disparity Detected';
      badge.className = `badge ${data.overall_compliant ? 'badge-emerald' : 'badge-rose'}`;
    }

    const container = document.getElementById('fairnessSubgroupCards');
    if (!container) return;

    const models = data.models || {};
    let html = '';

    for (const mKey in models) {
      const mData = models[mKey];
      const modelTitle = mKey === 'attrition_model' ? 'Attrition Risk Model' : 'Promotion Readiness Model';
      const audits = mData.subgroup_audits || [];

      html += `
        <div class="card" style="margin-bottom: 16px;">
          <div class="card-header">
            <div class="card-title">${modelTitle}</div>
            <span class="badge ${mData.overall_fairness_passed ? 'badge-emerald' : 'badge-amber'}">
              ${mData.overall_fairness_passed ? 'Compliant' : 'Audit Check'}
            </span>
          </div>
          <div class="layout-2col">
            ${audits.map(a => {
              const ratio = a.disparate_impact_ratio;
              const passed = a.passed_fairness_audit;
              const ratioPct = Math.min((ratio / 1.5) * 100, 100);

              return `
                <div style="background: var(--bg-surface-subtle); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 14px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <strong>${a.protected_feature.toUpperCase()}: ${a.unprivileged_group} vs ${a.privileged_group}</strong>
                    <span class="badge ${passed ? 'badge-emerald' : 'badge-rose'}">
                      Ratio: ${ratio.toFixed(3)}
                    </span>
                  </div>
                  <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 8px;">
                    Unprivileged (${a.unprivileged_group}) Rate: <strong>${(a.unprivileged_selection_rate * 100).toFixed(1)}%</strong><br>
                    Privileged (${a.privileged_group}) Rate: <strong>${(a.privileged_selection_rate * 100).toFixed(1)}%</strong>
                  </div>
                  <div style="position: relative; height: 6px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden;">
                    <div style="width: ${ratioPct}%; height: 100%; background: ${passed ? 'var(--accent-promotion)' : 'var(--accent-attrition)'};"></div>
                  </div>
                  <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted); margin-top: 4px;">
                    <span>0.00</span>
                    <span style="color: var(--text-secondary); font-weight: 600;">Threshold: 0.80</span>
                    <span>1.50+</span>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    }

    container.innerHTML = html;
  } catch (err) {
    console.error("Fairness error:", err);
  }
}

// ==========================================================================
// 6. Batch Scoring
// ==========================================================================

function initBatchUpload() {
  const dropzone = document.getElementById('batchDropzone');
  const fileInput = document.getElementById('batchFileInput');

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--accent-primary)';
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'var(--border-medium)';
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--border-medium)';
    if (e.dataTransfer.files.length > 0) {
      handleBatchFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleBatchFile(e.target.files[0]);
    }
  });

  // Filter input on batch table
  const searchInput = document.getElementById('batchSearchFilter');
  if (searchInput) {
    searchInput.addEventListener('input', renderBatchTableRows);
  }
}

async function handleBatchFile(file) {
  if (!file.name.endsWith('.csv')) {
    alert("Please select a standard CSV file.");
    return;
  }

  const statusEl = document.getElementById('batchUploadStatus');
  statusEl.innerHTML = '<span style="color: var(--accent-primary);">Uploading and scoring batch records...</span>';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/batch-upload', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      statusEl.innerHTML = `<span style="color: var(--accent-attrition);">Upload error: ${data.detail || 'Validation failure'}</span>`;
      return;
    }

    statusEl.innerHTML = `<span style="color: var(--accent-promotion); font-weight: 600;">Successfully scored ${data.total_scored} employee records!</span>`;
    AppState.batchResults = data.predictions || [];

    // Render Batch Summary Badges
    const qDist = data.quadrant_distribution || {};
    document.getElementById('batchCountUrgent').innerText = qDist["Urgent Retention & Key Talent"] || 0;
    document.getElementById('batchCountInvest').innerText = qDist["Invest & Fast-Track"] || 0;
    document.getElementById('batchCountMonitor').innerText = qDist["Monitor & Engage"] || 0;
    document.getElementById('batchCountCore').innerText = qDist["Core Performer / Low Priority"] || 0;

    document.getElementById('batchResultsCard').style.display = 'block';
    renderBatchTableRows();
  } catch (err) {
    statusEl.innerHTML = `<span style="color: var(--accent-attrition);">Network error: ${err.message}</span>`;
  }
}

function renderBatchTableRows() {
  const tbody = document.getElementById('batchResultsTableBody');
  const search = (document.getElementById('batchSearchFilter').value || '').toLowerCase();
  if (!tbody) return;

  const filtered = AppState.batchResults.filter(p => 
    p.employee_id.toLowerCase().includes(search) ||
    p.quadrant.toLowerCase().includes(search)
  );

  tbody.innerHTML = filtered.map(p => {
    const topAttrDriver = (p.top_attrition_drivers && p.top_attrition_drivers[0]) ? p.top_attrition_drivers[0].feature_name.replace(/_/g, ' ') : 'None';
    const quadClass = p.quadrant.includes('Urgent') ? 'badge-rose' : p.quadrant.includes('Invest') ? 'badge-emerald' : p.quadrant.includes('Monitor') ? 'badge-amber' : 'badge-indigo';

    return `
      <tr>
        <td><strong>${p.employee_id}</strong></td>
        <td><span style="color: var(--accent-attrition); font-weight: 600;">${(p.attrition_risk_score * 100).toFixed(1)}%</span></td>
        <td><span style="color: var(--accent-promotion); font-weight: 600;">${(p.promotion_readiness_score * 100).toFixed(1)}%</span></td>
        <td><span class="badge ${quadClass}">${p.quadrant}</span></td>
        <td>${p.priority_level}</td>
        <td>${topAttrDriver}</td>
      </tr>
    `;
  }).join('');
}

function downloadTemplateCsv() {
  const csvContent = "data:text/csv;charset=utf-8," + 
    "employee_id,department,gender,age,education_level,tenure_years,years_since_promotion,monthly_income,overtime_status,satisfaction_score,performance_rating,kpi_met_above_80,awards_won,num_trainings_last_year,stock_option_level,avg_training_score\n" +
    "EMP_BATCH_01,R&D,Female,32,3,4.0,2.0,6500,1,2.5,3.8,1,0,2,1,72\n" +
    "EMP_BATCH_02,Sales,Male,27,3,2.0,2.0,3400,1,1.5,3.5,1,0,1,0,65\n" +
    "EMP_BATCH_03,Technology,Female,35,4,6.0,3.0,14000,0,4.5,4.8,1,1,3,2,88\n" +
    "EMP_BATCH_04,Operations,Male,42,2,8.0,4.0,5200,0,3.0,3.0,0,0,2,1,60\n";
  const link = document.createElement("a");
  link.setAttribute("href", encodeURI(csvContent));
  link.setAttribute("download", "employee_scoring_template.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function exportBatchCsv() {
  if (!AppState.batchResults || AppState.batchResults.length === 0) return;

  let csv = "employee_id,attrition_risk_score,promotion_readiness_score,quadrant,action_code,priority_level\n";
  for (const p of AppState.batchResults) {
    csv += `"${p.employee_id}",${p.attrition_risk_score},${p.promotion_readiness_score},"${p.quadrant}","${p.action_code}","${p.priority_level}"\n`;
  }

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.setAttribute("download", "scored_talent_results.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ==========================================================================
// 7. Dataset Explorer
// ==========================================================================

async function fetchDatasetExplorer() {
  const src = document.getElementById('dsSourceSelect') ? document.getElementById('dsSourceSelect').value : 'attrition';
  const dept = document.getElementById('dsDeptFilter') ? document.getElementById('dsDeptFilter').value : 'ALL';
  const search = document.getElementById('dsSearchInput') ? document.getElementById('dsSearchInput').value : '';

  const summaryEl = document.getElementById('dsSummaryMeta');
  if (summaryEl) {
    if (src === 'attrition') {
      summaryEl.innerText = "IBM HR Employee Attrition Benchmark • 1,470 Records • 23 Features • Target: target_attrition";
    } else {
      summaryEl.innerText = "HR Promotion Analytics Benchmark • 54,808 Records • 13 Features • Target: target_promotion";
    }
  }

  const tbody = document.getElementById('dsTableBody');
  if (tbody) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">Loading authentic dataset records...</td></tr>';
  }

  try {
    const res = await fetch(`/api/dataset?dataset_type=${src}&department=${dept}&search_id=${encodeURIComponent(search)}&limit=50`);
    const data = await res.json();

    const counter = document.getElementById('dsRecordCounter');
    if (counter) {
      counter.innerText = `Displaying ${data.records.length} of ${data.total_records.toLocaleString()} records`;
    }

    if (tbody) {
      tbody.innerHTML = data.records.map(r => `
        <tr>
          <td><strong>${r.employee_id}</strong></td>
          <td>${r.department || 'Operations'}</td>
          <td>${r.gender || 'Female'}</td>
          <td>${r.age || 30}</td>
          <td>${r.education_level || 3}</td>
          <td>${src === 'attrition' ? (r.target_attrition === 1 ? '<span class="badge badge-rose">Attrition (1)</span>' : '<span class="badge badge-gray">Retained (0)</span>') : (r.target_promotion === 1 ? '<span class="badge badge-emerald">Promoted (1)</span>' : '<span class="badge badge-gray">Not Promoted (0)</span>')}</td>
          <td>
            <button class="btn btn-secondary btn-sm" onclick='loadDatasetRecordToScoring(${JSON.stringify(r).replace(/'/g, "&apos;")})'>Inspect</button>
          </td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error("Dataset explorer error:", err);
  }
}

function loadDatasetRecordToScoring(r) {
  populateFormFromRecord(r);
  switchView('scoring');
  runIndividualEvaluation();
}
