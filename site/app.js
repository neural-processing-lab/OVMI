(() => {
  "use strict";

  const GROUPS = [
    { key: "attempted_invasive", label: "Attempted speech / invasive" },
    { key: "perceived_noninvasive", label: "Perceived speech / non-invasive" }
  ];
  const state = {
    data: null,
    reference: "subtlex",
    group: "all",
    sort: "ovmi-desc"
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatNumber(value, digits = 3) {
    return Number(value).toFixed(digits);
  }

  function formatPercent(value, digits = 1) {
    return `${(100 * Number(value)).toFixed(digits)}%`;
  }

  function formatVocabulary(value) {
    return Number(value) >= 1000
      ? `${Math.round(Number(value) / 1000)}k`
      : Number(value).toLocaleString("en-GB");
  }

  function groupLabel(key) {
    return GROUPS.find((group) => group.key === key)?.label || key;
  }

  function currentResult(system) {
    return system.references[state.reference];
  }

  function applyConfigLinks() {
    const config = window.OVMI_SITE_CONFIG || {};
    const keyMap = {
      paper: "paperUrl",
      package: "packageUrl",
      blog: "blogUrl",
      repository: "repositoryUrl",
      addResult: "addResultUrl"
    };
    $$('[data-config-link]').forEach((link) => {
      const key = keyMap[link.dataset.configLink];
      const configured = config[key];
      if (configured) {
        link.href = configured;
        if (/^https?:/.test(configured)) {
          link.rel = "noopener noreferrer";
        }
      } else {
        link.href = config.unavailableLinkTarget || "#project-links";
        link.title = "Public link forthcoming";
        link.setAttribute("aria-describedby", `${link.dataset.configLink}-status`);
      }
    });
  }

  function metricHtml(system) {
    const metric = system.metric;
    if (metric.type === "wer") {
      const bridge = metric.p_is_lower_bound ? "P ≥" : "P =";
      return `${formatPercent(metric.reported_value)} WER` +
        `<span class="cell-note metric-lower-bound">${bridge} ${formatPercent(metric.p_correct)} conservative lower bound</span>`;
    }
    return `${formatPercent(metric.reported_value)}<span class="cell-note">${escapeHtml(metric.label)}</span>`;
  }

  function uncertaintyHtml(result) {
    const uncertainty = result.uncertainty;
    if (!uncertainty) return '<span class="cell-note">Not reported</span>';
    if (uncertainty.kind === "seed_sem") {
      const delta = Math.max(
        result.ovmi_bits - uncertainty.low_bits,
        uncertainty.high_bits - result.ovmi_bits
      );
      return `± ${formatNumber(delta)} bits<span class="cell-note">one SEM across seeds</span>`;
    }
    return `${formatNumber(uncertainty.low_bits)}–${formatNumber(uncertainty.high_bits)}<span class="cell-note">${escapeHtml(uncertainty.label)}</span>`;
  }

  function sortSystems(systems) {
    const [key, direction] = state.sort.split("-");
    const multiplier = direction === "asc" ? 1 : -1;
    return [...systems].sort((left, right) => {
      let a;
      let b;
      if (key === "ovmi") {
        a = currentResult(left).ovmi_bits;
        b = currentResult(right).ovmi_bits;
      } else if (key === "percent") {
        a = currentResult(left).ovmi_percent;
        b = currentResult(right).ovmi_percent;
      } else if (key === "vocabulary") {
        a = left.vocabulary_size;
        b = right.vocabulary_size;
      } else {
        a = left[key];
        b = right[key];
      }
      if (typeof a === "string") return multiplier * a.localeCompare(b);
      if (a === b) return left.system.localeCompare(right.system);
      return multiplier * (a - b);
    });
  }

  function renderLeaderboard() {
    const tbody = $("#leaderboard-body");
    tbody.textContent = "";
    const groups = state.group === "all"
      ? GROUPS
      : GROUPS.filter((group) => group.key === state.group);

    groups.forEach((group) => {
      const groupSystems = state.data.systems.filter((system) => system.group === group.key);
      const visible = sortSystems(groupSystems);
      const groupRow = document.createElement("tr");
      groupRow.className = "group-row";
      groupRow.innerHTML = `<th colspan="9" scope="rowgroup">${escapeHtml(group.label)}</th>`;
      tbody.append(groupRow);

      visible.forEach((system) => {
        const result = currentResult(system);
        const row = document.createElement("tr");
        const setting = system.group === "attempted_invasive" ? "Attempted" : "Perceived";
        const decoderNote = system.decoder_method
          ? `<span class="cell-note decoder-method">Decoder: ${escapeHtml(system.decoder_method)}</span>`
          : "";
        row.innerHTML = `
          <td><span class="system-name">${escapeHtml(system.system)}</span><span class="system-task">${escapeHtml(system.task)}</span>${decoderNote}</td>
          <td>${system.year}</td>
          <td>${setting}<span class="cell-note">${escapeHtml(system.modality)}</span></td>
          <td>${formatVocabulary(system.vocabulary_size)}</td>
          <td>${metricHtml(system)}</td>
          <td class="score-cell">${formatNumber(result.ovmi_bits)}</td>
          <td class="score-cell">${formatNumber(result.ovmi_percent, 1)}%</td>
          <td>${uncertaintyHtml(result)}</td>
          <td><a href="${escapeHtml(system.source.primary_url)}" rel="noopener noreferrer">${escapeHtml(system.source.citation)}</a></td>`;
        tbody.append(row);
      });
    });
  }

  function renderReferenceState() {
    const reference = state.data.references[state.reference];
    $("#reference-entropy").value = `${formatNumber(reference.entropy_bits, 2)} bits`;
    $("#reference-description").textContent = reference.description;
    $$(".current-reference-label").forEach((node) => {
      node.textContent = `Reference: ${reference.short_label}.`;
    });
    const willett = state.data.systems.find((system) => system.id === "willett_2023_v50-w");
    if (willett) {
      $("#willett-broad").textContent = `${formatNumber(willett.references.subtlex.ovmi_percent, 1)}%`;
      $("#willett-ucv").textContent = `${formatNumber(willett.references.ucv.ovmi_percent, 1)}%`;
    }
  }

  function renderAll() {
    renderReferenceState();
    renderLeaderboard();
  }

  function bindControls() {
    $$('#reference-selector input[name="reference"]').forEach((input) => {
      input.addEventListener("change", (event) => {
        state.reference = event.target.value;
        renderAll();
      });
    });
    $("#group-filter").addEventListener("change", (event) => {
      state.group = event.target.value;
      renderLeaderboard();
    });
    $("#sort-select").addEventListener("change", (event) => {
      state.sort = event.target.value;
      renderLeaderboard();
    });
  }

  async function initialise() {
    applyConfigLinks();
    bindControls();
    try {
      const response = await fetch("./data/leaderboard.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.data = await response.json();
      renderAll();
    } catch (error) {
      const tbody = $("#leaderboard-body");
      tbody.innerHTML = `<tr><td colspan="10" class="loading-cell">Could not load benchmark data. Preview the site through a local web server rather than opening index.html directly. (${escapeHtml(error.message)})</td></tr>`;
      console.error("Failed to load OVMI site data", error);
    }
  }

  initialise();
})();
