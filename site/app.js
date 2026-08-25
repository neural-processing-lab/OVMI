(() => {
  "use strict";

  const GROUPS = [
    { key: "attempted_invasive", label: "Attempted speech / invasive" },
    { key: "perceived_noninvasive", label: "Perceived speech / non-invasive" }
  ];
  const state = {
    data: null,
    reference: "subtlex",
    group: "attempted_invasive",
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
      const sem = metric.reported_sem
        ? `<span class="cell-note">± ${formatPercent(metric.reported_sem, 2)} SEM across participants</span>`
        : "";
      return `${formatPercent(metric.reported_value)} WER` +
        sem + `<span class="cell-note metric-lower-bound">${bridge} ${formatPercent(metric.p_correct)} conservative lower bound</span>`;
    }
    return `${formatPercent(metric.reported_value)}<span class="cell-note">${escapeHtml(metric.label)}</span>`;
  }

  function uncertaintyHtml(result) {
    const uncertainty = result.uncertainty;
    if (!uncertainty) return '<span class="cell-note">Not reported</span>';
    if (uncertainty.kind === "seed_sem" || uncertainty.kind === "participant_sem") {
      const delta = Math.max(
        result.ovmi_bits - uncertainty.low_bits,
        uncertainty.high_bits - result.ovmi_bits
      );
      return `± ${formatNumber(delta)} bits<span class="cell-note">${escapeHtml(uncertainty.label)}</span>`;
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

  function comparisonMetric(system) {
    const [key] = state.sort.split("-");
    const result = currentResult(system);
    const entropy = state.data.references[state.reference].entropy_bits;
    if (key === "percent") {
      return {
        label: "OVMI / H(p)",
        value: result.ovmi_percent,
        maximum: 100,
        uncertainty: result.uncertainty && {
          low: 100 * result.uncertainty.low_bits / entropy,
          high: 100 * result.uncertainty.high_bits / entropy
        }
      };
    }
    if (key === "year") {
      const years = state.data.systems.map((item) => item.year);
      return {
        label: "Publication year",
        value: system.year,
        minimum: Math.min(...years),
        maximum: Math.max(...years)
      };
    }
    if (key === "vocabulary") {
      return {
        label: "Vocabulary size",
        value: system.vocabulary_size,
        maximum: Math.max(...state.data.systems.map((item) => item.vocabulary_size))
      };
    }
    return {
      label: "OVMI bits",
      value: result.ovmi_bits,
      maximum: entropy,
      uncertainty: result.uncertainty && {
        low: result.uncertainty.low_bits,
        high: result.uncertainty.high_bits
      }
    };
  }

  function comparisonBarHtml(system) {
    const metric = comparisonMetric(system);
    const range = metric.maximum - (metric.minimum || 0);
    const position = range ? 100 * (metric.value - (metric.minimum || 0)) / range : 100;
    const uncertainty = metric.uncertainty;
    const interval = uncertainty
      ? `<span class="score-interval" style="left:${Math.max(0, 100 * (uncertainty.low - (metric.minimum || 0)) / range)}%;width:${Math.max(1, 100 * (uncertainty.high - uncertainty.low) / range)}%"></span>`
      : "";
    const value = metric.label === "OVMI / H(p)"
      ? `${formatNumber(metric.value, 1)}%`
      : metric.label === "Vocabulary size"
        ? formatVocabulary(metric.value)
        : metric.label === "Publication year"
          ? String(metric.value)
          : `${formatNumber(metric.value)} bits`;
    return `<div class="comparison-bar" aria-label="${escapeHtml(metric.label)}: ${escapeHtml(value)}">
      <span class="comparison-track"><span class="comparison-fill" style="width:${Math.max(1, Math.min(100, position))}%"></span>${interval}</span>
      <span class="comparison-value">${escapeHtml(value)}</span>
    </div>`;
  }

  function pointMark(system, index, x, y, colour) {
    const title = `${system.system}: ${formatNumber(currentResult(system).ovmi_percent, 1)}% of the selected reference`;
    const mark = index % 4;
    const base = `<title>${escapeHtml(title)}</title>`;
    if (mark === 1) return `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" fill="${colour}" stroke="#ffffff" stroke-width="1.4">${base}</rect>`;
    if (mark === 2) return `<path d="M ${x} ${y - 5} L ${x + 5} ${y + 4} L ${x - 5} ${y + 4} Z" fill="${colour}" stroke="#ffffff" stroke-width="1.4">${base}</path>`;
    if (mark === 3) return `<path d="M ${x} ${y - 5} L ${x + 5} ${y} L ${x} ${y + 5} L ${x - 5} ${y} Z" fill="${colour}" stroke="#ffffff" stroke-width="1.4">${base}</path>`;
    return `<circle cx="${x}" cy="${y}" r="4.5" fill="${colour}" stroke="#ffffff" stroke-width="1.4">${base}</circle>`;
  }

  function plotLabel(system) {
    const duplicateCount = state.data.systems.filter((item) => item.system === system.system).length;
    return duplicateCount > 1 ? `${system.system} ${formatVocabulary(system.vocabulary_size)}` : system.system;
  }

  function nicePlotMaximum(value) {
    const magnitude = 10 ** Math.floor(Math.log10(Math.max(value, 0.01)));
    const normalised = value / magnitude;
    const step = normalised <= 1 ? 1 : normalised <= 2 ? 2 : normalised <= 2.5 ? 2.5 : normalised <= 5 ? 5 : 10;
    return step * magnitude;
  }

  function formatPlotTick(value) {
    return `${Number(value.toFixed(value < 10 ? 2 : 0))}%`;
  }

  function renderBenchmarkScatter() {
    const container = $("#benchmark-scatter");
    const width = 1120;
    const height = 282;
    const panelWidth = 520;
    const panelGap = 48;
    const margin = { top: 38, right: 18, bottom: 38, left: 42 };
    const years = state.data.systems.map((system) => system.year);
    const minYear = Math.min(...years);
    const maxYear = Math.max(...years);
    const yearSpan = Math.max(1, maxYear - minYear);
    const panels = [
      { group: GROUPS[0], x: 0, colour: "#46515d" },
      { group: GROUPS[1], x: panelWidth + panelGap, colour: "#176b66" }
    ];
    const x = (year, panelX) => panelX + margin.left + ((year - minYear) / yearSpan) * (panelWidth - margin.left - margin.right);
    const parts = [`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Published systems by year and normalised OVMI for invasive and non-invasive study groups"><title>Published systems by year and normalised OVMI</title>`];
    panels.forEach((panel) => {
      const panelSystems = state.data.systems.filter((system) => system.group === panel.group.key);
      const yMaximum = nicePlotMaximum(Math.max(...panelSystems.map((system) => currentResult(system).ovmi_percent)));
      const y = (value) => margin.top + (1 - value / yMaximum) * (height - margin.top - margin.bottom);
      parts.push(`<text class="scatter-panel-title" x="${panel.x + margin.left}" y="18">${escapeHtml(panel.group.label)}</text>`);
      [0, yMaximum / 2, yMaximum].forEach((tick) => {
        const yPos = y(tick);
        parts.push(`<line class="scatter-grid" x1="${panel.x + margin.left}" x2="${panel.x + panelWidth - margin.right}" y1="${yPos}" y2="${yPos}"/>`);
        parts.push(`<text class="scatter-tick" x="${panel.x + margin.left - 8}" y="${yPos + 3}" text-anchor="end">${formatPlotTick(tick)}</text>`);
      });
      for (let year = minYear; year <= maxYear; year += 1) {
        const xPos = x(year, panel.x);
        parts.push(`<line class="scatter-axis" x1="${xPos}" x2="${xPos}" y1="${height - margin.bottom}" y2="${height - margin.bottom + 4}"/>`);
        parts.push(`<text class="scatter-tick" x="${xPos}" y="${height - 14}" text-anchor="middle">${year}</text>`);
      }
      parts.push(`<line class="scatter-axis" x1="${panel.x + margin.left}" x2="${panel.x + margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"/>`);
      parts.push(`<line class="scatter-axis" x1="${panel.x + margin.left}" x2="${panel.x + panelWidth - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"/>`);
      panelSystems.forEach((system, index) => {
        const xPos = x(system.year, panel.x);
        const yPos = y(currentResult(system).ovmi_percent);
        parts.push(`<g class="scatter-point">${pointMark(system, index, xPos, yPos, panel.colour)}<text x="${xPos + 7}" y="${yPos - 7}">${escapeHtml(plotLabel(system))}</text></g>`);
      });
    });
    parts.push(`<text class="scatter-axis-label" x="${width / 2}" y="${height - 1}" text-anchor="middle">Publication year</text>`);
    parts.push("</svg>");
    container.innerHTML = parts.join("");
  }

  function renderLeaderboard() {
    const tbody = $("#leaderboard-body");
    tbody.textContent = "";
    const groups = state.group === "all"
      ? GROUPS
      : GROUPS.filter((group) => group.key === state.group);
    $("#bar-metric-label").textContent = comparisonMetric(state.data.systems[0]).label;

    groups.forEach((group) => {
      const groupSystems = state.data.systems.filter((system) => system.group === group.key);
      const visible = sortSystems(groupSystems);
      const groupRow = document.createElement("tr");
      groupRow.className = "group-row";
      groupRow.innerHTML = `<th colspan="10" scope="rowgroup">${escapeHtml(group.label)}</th>`;
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
          <td>${comparisonBarHtml(system)}</td>
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
    renderBenchmarkScatter();
    renderLeaderboard();
  }

  function bindControls() {
    $$('#reference-selector input[name="reference"]').forEach((input) => {
      input.addEventListener("change", (event) => {
        state.reference = event.target.value;
        renderAll();
      });
    });
    $$('#group-filter input[name="group"]').forEach((input) => {
      input.addEventListener("change", (event) => {
        state.group = event.target.value;
        renderLeaderboard();
      });
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
      tbody.innerHTML = `<tr><td colspan="10" class="loading-cell">Could not load comparison data. Preview the site through a local web server rather than opening index.html directly. (${escapeHtml(error.message)})</td></tr>`;
      console.error("Failed to load OVMI site data", error);
    }
  }

  initialise();
})();
