(() => {
      const data = window.PAIRWISE_DATA;
      const root = document.getElementById("root");
      const storageBase = "pairwise-html:" + data.meta.storage_key + ":";
      const P_INTENSITY_FLOOR = 1e-12;
      const state = {
        view: loadState("view", "matrix"),
        regex: loadState("regex", ""),
        alpha: 0.05,
        matrixSort: "strength",
        anchorIndex: null,
        tableSortKey: "avg",
        tableSortDir: "desc",
        hiddenRows: new Set(),
        hiddenCols: new Set(),
      };
      let hoverPair = null;

      function loadState(key, fallback) {
        const value = window.localStorage.getItem(storageBase + key);
        return value === null ? fallback : value;
      }

      function persistState() {
        window.localStorage.setItem(storageBase + "view", state.view);
        window.localStorage.setItem(storageBase + "regex", state.regex);
      }

      function esc(value) {
        return String(value ?? "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }

      function isNumber(value) {
        return typeof value === "number" && Number.isFinite(value);
      }

      function fmtPct(value, digits = 1) {
        if (!isNumber(value)) return "-";
        return (value * 100).toFixed(digits);
      }

      function fmtDiff(value, digits = 1) {
        if (!isNumber(value)) return "-";
        const rendered = (value * 100).toFixed(digits);
        return value > 0 ? "+" + rendered : rendered;
      }

      function fmtPp(value, digits = 1) {
        if (!isNumber(value)) return "—";
        return fmtPct(value, digits) + " pp";
      }

      function fmtP(value) {
        if (!isNumber(value)) return "-";
        if (value < 0.001) return "<0.001";
        return value.toFixed(3);
      }

      function scoreDisplayFormat(meta) {
        return meta?.score_display_format || "percentage";
      }

      function scoreUnit(meta) {
        return meta?.score_unit ?? null;
      }

      function scoreHigherIsBetter(meta) {
        return meta?.higher_is_better !== false;
      }

      function isPercentageMetric(meta) {
        return scoreDisplayFormat(meta) === "percentage";
      }

      function fmtScore(value, meta, digits = 1) {
        if (!isNumber(value)) return "-";
        if (isPercentageMetric(meta)) return fmtPct(value, digits);
        return Number(value).toFixed(digits);
      }

      function fmtScoreDiff(value, meta, digits = 1) {
        if (!isNumber(value)) return "-";
        if (isPercentageMetric(meta)) return fmtDiff(value, digits);
        const rendered = Number(value).toFixed(digits);
        return value > 0 ? "+" + rendered : rendered;
      }

      function fmtScoreValue(value, meta, digits = 1) {
        if (!isNumber(value)) return "-";
        return isPercentageMetric(meta)
          ? `${fmtScore(value, meta, digits)}%`
          : fmtScore(value, meta, digits);
      }

      function fmtDelta(value, meta, digits = 1) {
        if (!isNumber(value)) return "—";
        return isPercentageMetric(meta)
          ? `${fmtScoreDiff(value, meta, digits)} pp`
          : fmtScoreDiff(value, meta, digits);
      }

      function fmtMde(value, meta, digits = 1) {
        if (!isNumber(value)) return "—";
        return isPercentageMetric(meta)
          ? `${fmtPct(value, digits)} pp`
          : fmtScore(value, meta, digits);
      }

      function comparisonValue(value, meta) {
        if (!isNumber(value)) return null;
        return scoreHigherIsBetter(meta) ? value : -value;
      }

      function fmtScaleP(value) {
        if (!isNumber(value) || value <= 0) return "0";
        if (value >= 0.001) return value.toFixed(3);
        if (value < 1e-4) return value.toExponential(0).replace("e+", "e");
        const digits = Math.min(20, Math.max(3, Math.ceil(-Math.log10(value))));
        return value
          .toLocaleString("en-US", {
            useGrouping: false,
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
          })
          .replace(/0+$/, "")
          .replace(/[.]$/, "");
      }

      function matrixMde(meta, alpha) {
        if (!meta) return null;
        const byAlpha = meta.mde80_by_alpha;
        if (
          byAlpha &&
          Object.prototype.hasOwnProperty.call(byAlpha, String(alpha)) &&
          isNumber(byAlpha[String(alpha)])
        ) {
          return byAlpha[String(alpha)];
        }
        return isNumber(meta.mde80) ? meta.mde80 : null;
      }

      function sortArrow(key, sortState = null) {
        const activeKey = sortState?.key ?? state.tableSortKey;
        const activeDir = sortState?.dir ?? state.tableSortDir;
        const stateClass = activeKey !== key
          ? "is-idle"
          : activeDir === "asc"
            ? "is-asc"
            : "is-desc";
        return `<span class="sort-glyph ${stateClass}">${sortSvg()}</span>`;
      }

      function compileRegex() {
        if (!state.regex) return { regex: null, error: false };
        try {
          return { regex: new RegExp(state.regex, "i"), error: false };
        } catch (_error) {
          return { regex: null, error: true };
        }
      }

      function filteredModelIndices() {
        const compiled = compileRegex();
        const all = data.models.map((model) => model.index);
        if (compiled.error || !compiled.regex) {
          return { indices: all, error: compiled.error };
        }
        return {
          indices: all.filter((index) =>
            compiled.regex.test(data.models[index].display_label)
          ),
          error: false,
        };
      }

      function displayScore(model) {
        if (isNumber(model.display_score)) return model.display_score;
        if (isNumber(model.avg_task_score)) return model.avg_task_score;
        if (isNumber(model.shared_score)) return model.shared_score;
        return null;
      }

      function comparisonDiff(row, col) {
        const diff = data.matrix.score_diff[row]?.[col];
        return comparisonValue(diff, data.meta);
      }

      function pairDirection(row, col) {
        const diff = comparisonDiff(row, col);
        if (isNumber(diff) && diff !== 0) return Math.sign(diff);
        const winRate = data.matrix.win_rate[row]?.[col];
        if (!isNumber(winRate)) return 0;
        if (winRate > 0.5) return 1;
        if (winRate < 0.5) return -1;
        return 0;
      }

      function cellColor(direction, pValue, alpha) {
        if (direction === 0 || !isNumber(pValue)) {
          return { bg: "var(--c-neutral-50)", fg: "var(--c-ink-60)" };
        }
        const significant = pValue <= alpha;
        if (!significant) {
          const hue = direction > 0 ? 150 : 25;
          return {
            bg: "oklch(0.96 0.01 " + hue + ")",
            fg: "var(--c-ink-60)",
            border: "var(--c-rule)",
          };
        }
        const hue = direction > 0 ? 150 : 25;
        const clampedP = Math.max(P_INTENSITY_FLOOR, Math.min(alpha, pValue));
        const raw = (Math.log10(alpha) - Math.log10(clampedP)) /
          (Math.log10(alpha) - Math.log10(P_INTENSITY_FLOOR));
        const t = Math.sqrt(Math.max(0, Math.min(1, raw)));
        const lightness = (0.92 - 0.25 * t).toFixed(3);
        const chroma = (0.06 + 0.11 * t).toFixed(3);
        const fg = lightness < 0.72 ? "var(--c-paper)" : "var(--c-ink-70)";
        return {
          bg: "oklch(" + lightness + " " + chroma + " " + hue + ")",
          fg,
          border: "transparent",
        };
      }

      function cellSignalLevel(pValue, alpha) {
        if (!isNumber(pValue) || pValue > alpha) return 0;
        if (pValue <= 0.001) return 3;
        if (pValue <= 0.01) return 2;
        return 1;
      }

      function renderCellSignal(level) {
        if (level <= 0) return "";
        return `<span class="cell-signal sig-${level}" aria-hidden="true"></span>`;
      }

      function matrixOrder(indices) {
        const ordered = indices.slice();
        if (state.matrixSort === "name") {
          ordered.sort((a, b) =>
            data.models[a].display_label.localeCompare(data.models[b].display_label)
          );
          return ordered;
        }
        if (state.matrixSort === "score") {
          ordered.sort((a, b) => {
            const av = comparisonValue(displayScore(data.models[a]), data.meta);
            const bv = comparisonValue(displayScore(data.models[b]), data.meta);
            return (bv ?? -Infinity) - (av ?? -Infinity);
          });
          return ordered;
        }
        if (state.matrixSort === "anchor" && indices.includes(state.anchorIndex)) {
          const anchor = state.anchorIndex;
          ordered.sort((a, b) => {
            if (a === anchor) return -1;
            if (b === anchor) return 1;
            const av = comparisonDiff(a, anchor);
            const bv = comparisonDiff(b, anchor);
            const da = isNumber(av) ? av : data.matrix.win_rate[a]?.[anchor] ?? 0.5;
            const db = isNumber(bv) ? bv : data.matrix.win_rate[b]?.[anchor] ?? 0.5;
            return db - da;
          });
          return ordered;
        }
        ordered.sort((a, b) => {
          const av = data.models[a].strength ?? -Infinity;
          const bv = data.models[b].strength ?? -Infinity;
          if (bv !== av) return bv - av;
          return (
            comparisonValue(displayScore(data.models[b]), data.meta) ?? -Infinity
          ) - (
            comparisonValue(displayScore(data.models[a]), data.meta) ?? -Infinity
          );
        });
        return ordered;
      }

      function summaryFor(modelIndex, indices) {
        let wins = 0;
        let losses = 0;
        let ties = 0;
        indices.forEach((other) => {
          if (other === modelIndex) return;
          const pValue = data.matrix.p_value[modelIndex]?.[other];
          const direction = pairDirection(modelIndex, other);
          if (isNumber(pValue) && pValue <= state.alpha) {
            if (direction > 0) wins += 1;
            else if (direction < 0) losses += 1;
            else ties += 1;
          } else {
            ties += 1;
          }
        });
        return { wins, losses, ties };
      }

      function visibleRows(indices) {
        return indices.filter((index) => !state.hiddenRows.has(index));
      }

      function visibleTaskColumns() {
        return data.task_columns.filter((column) => !state.hiddenCols.has(column.id));
      }

      function columnsComparable(columns) {
        if (columns.length === 0) return false;
        const format = scoreDisplayFormat(columns[0]);
        const unit = scoreUnit(columns[0]);
        const higherIsBetter = scoreHigherIsBetter(columns[0]);
        return columns.every((column) =>
          scoreDisplayFormat(column) === format &&
          scoreUnit(column) === unit &&
          scoreHigherIsBetter(column) === higherIsBetter
        );
      }

      function aggregateColumnMeta(columns) {
        if (!columnsComparable(columns)) return null;
        return {
          score_display_format: scoreDisplayFormat(columns[0]),
          score_unit: scoreUnit(columns[0]),
          higher_is_better: scoreHigherIsBetter(columns[0]),
        };
      }

      function defaultScoreSortDir(meta) {
        return scoreHigherIsBetter(meta) ? "desc" : "asc";
      }

      function showAverageColumn(columns) {
        return columns.length > 1 && columnsComparable(columns);
      }

      function resolvedTableSort(columns, showAverage) {
        if (state.tableSortKey === "name") {
          return { key: "name", dir: state.tableSortDir };
        }
        if (state.tableSortKey === "avg") {
          if (showAverage) return { key: "avg", dir: state.tableSortDir };
          if (columns[0]) return { key: columns[0].id, dir: defaultScoreSortDir(columns[0]) };
          return { key: "name", dir: "asc" };
        }
        if (columns.some((column) => column.id === state.tableSortKey)) {
          return { key: state.tableSortKey, dir: state.tableSortDir };
        }
        if (showAverage) return { key: "avg", dir: defaultScoreSortDir(aggregateColumnMeta(columns)) };
        if (columns[0]) return { key: columns[0].id, dir: defaultScoreSortDir(columns[0]) };
        return { key: "name", dir: "asc" };
      }

      function averageVisibleScore(model, columns) {
        if (!columnsComparable(columns)) return null;
        const scores = columns
          .map((column) => model.task_scores[column.id])
          .filter((value) => isNumber(value));
        if (scores.length > 0) {
          return scores.reduce((sum, value) => sum + value, 0) / scores.length;
        }
        return displayScore(model);
      }

      function compareValues(a, b, direction) {
        const order = direction === "asc" ? 1 : -1;
        if (a === b) return 0;
        if (a === null || a === undefined) return 1;
        if (b === null || b === undefined) return -1;
        if (typeof a === "string" || typeof b === "string") {
          return a < b ? -order : order;
        }
        return (a - b) * order;
      }

      function sortedTableRows(indices, columns, sortState) {
        const rows = visibleRows(indices).map((index) => data.models[index]);
        rows.sort((left, right) => {
          if (sortState.key === "name") {
            return compareValues(
              left.display_label,
              right.display_label,
              sortState.dir
            );
          }
          if (sortState.key === "avg") {
            return compareValues(
              averageVisibleScore(left, columns),
              averageVisibleScore(right, columns),
              sortState.dir
            );
          }
          return compareValues(
            left.task_scores[sortState.key],
            right.task_scores[sortState.key],
            sortState.dir
          );
        });
        return rows;
      }

      function renderSelect(label, value) {
        return `
          <label class="select">
            <span class="select-label">${esc(label)}</span>
            <div class="select-wrap">
              <select disabled>
                <option>${esc(value)}</option>
              </select>
              <span class="select-caret">${caretSvg()}</span>
            </div>
          </label>
        `;
      }

      function renderTopBar(filtered, regexError) {
        const scopeLabel = data.meta.scope_label +
          "  ·  " +
          data.meta.model_count +
          " models";
        const metricLabel = data.meta.metric +
          "  ·  shared n=" +
          data.meta.shared_n;
        return `
          <header class="topbar">
            <div class="brand">
              <div class="brand-copy">
                <div class="brand-name">olmo-eval</div>
                <div class="brand-sub">Results viewer</div>
              </div>
              <div class="tabs" role="tablist">
                <button
                  class="tab ${state.view === "matrix" ? "active" : ""}"
                  data-action="view"
                  data-view="matrix"
                >paired test</button>
                <button
                  class="tab ${state.view === "table" ? "active" : ""}"
                  data-action="view"
                  data-view="table"
                >results</button>
              </div>
            </div>
            <div class="selectors ${state.view === "matrix" ? "selectors-stacked" : ""}">
              ${renderSelect("suite / task", scopeLabel)}
              ${renderSelect("scoring", metricLabel)}
            </div>
            <div class="filters">
              <div class="regex ${regexError ? "err" : ""}">
                <span class="regex-slash">/</span>
                <input
                  id="regex-filter"
                  class="regex-input"
                  value="${esc(state.regex)}"
                  placeholder="search model names..."
                  spellcheck="false"
                  autocapitalize="off"
                  autocomplete="off"
                />
                <span class="regex-slash">/</span>
                <span class="regex-count">
                  ${regexError ? "invalid" : filtered.length + "/" + data.models.length}
                </span>
              </div>
            </div>
          </header>
        `;
      }

      function renderMetaStrip(indices) {
        return `
          <div class="meta-strip">
            <span class="meta-chip">
              <span class="meta-label">models</span>
              <span class="meta-value">${indices.length}</span>
            </span>
            <span class="meta-chip">
              <span class="meta-label">shared n</span>
              <span class="meta-value">${data.meta.shared_n}</span>
            </span>
            <span class="meta-chip">
              <span class="meta-label">margin</span>
              <span class="meta-value">${data.meta.margin}</span>
            </span>
            <span class="meta-chip">
              <span class="meta-label">matched</span>
              <span class="meta-value">${data.meta.matched_experiments}</span>
            </span>
          </div>
        `;
      }

      function renderMatrix(indices) {
        if (indices.length === 0) {
          return `
            <div class="matrix-wrap">
              ${renderMetaStrip(indices)}
              ${emptyState()}
            </div>
          `;
        }
        if (indices.length === 1) {
          return `
            <div class="matrix-wrap">
              ${renderMetaStrip(indices)}
              <div class="single-model-note">
                <div>only one model is currently visible.</div>
                <div class="dim">show at least two models to run a paired test.</div>
              </div>
            </div>
          `;
        }

        const order = matrixOrder(indices);
        const cellSize = 40;
        const labelWidth = 240;
        const summaryWidth = 112;

        const headers = order
          .map((modelIndex, position) => {
            const model = data.models[modelIndex];
            return `
              <button
                class="col-hdr ${state.anchorIndex === modelIndex ? "anchored" : ""}"
                style="grid-column:${position + 2};"
                data-action="anchor"
                data-index="${modelIndex}"
                data-col-index="${modelIndex}"
                title="anchor on ${esc(model.display_label)}"
              >
                <span
                  class="matrix-hdr-hide col-hdr-hide"
                  data-action="toggle-row"
                  data-index="${modelIndex}"
                  title="hide model"
                >x</span>
                <span class="col-hdr-inner">
                  <span class="col-hdr-name">${esc(model.display_label)}</span>
                </span>
              </button>
            `;
          })
          .join("");

        const rows = order
          .map((modelIndex, rowNumber) => {
            const model = data.models[modelIndex];
            const summary = summaryFor(modelIndex, order);
            const score = displayScore(model);
            const cells = order
              .map((otherIndex, colNumber) => {
                if (modelIndex === otherIndex) {
                  return `
                    <div
                      class="cell diag"
                      style="grid-row:${rowNumber + 2};grid-column:${colNumber + 2};"
                    >
                      <span class="diag-dot"></span>
                    </div>
                  `;
                }
                const pValue = data.matrix.p_value[modelIndex]?.[otherIndex];
                const direction = pairDirection(modelIndex, otherIndex);
                const diff = data.matrix.score_diff[modelIndex]?.[otherIndex];
                const style = cellColor(direction, pValue, state.alpha);
                const diffLabel = fmtScoreDiff(diff, data.meta, 1);
                const signalLevel = cellSignalLevel(pValue, state.alpha);
                return `
                  <div
                    class="cell"
                    style="
                      grid-row:${rowNumber + 2};
                      grid-column:${colNumber + 2};
                      background:${style.bg};
                      color:${style.fg};
                      border-color:${style.border ?? "var(--c-rule)"};
                    "
                    data-row="${modelIndex}"
                    data-col="${otherIndex}"
                    data-row-index="${modelIndex}"
                    data-col-index="${otherIndex}"
                  >
                    <span class="cell-inner">
                      <span class="cell-diff">${diffLabel}</span>
                      ${renderCellSignal(signalLevel)}
                    </span>
                  </div>
                `;
              })
              .join("");
            return `
              <button
                class="row-hdr ${state.anchorIndex === modelIndex ? "anchored" : ""}"
                style="grid-row:${rowNumber + 2};"
                data-action="anchor"
                data-index="${modelIndex}"
                data-row-index="${modelIndex}"
                title="anchor on ${esc(model.display_label)}"
              >
                <span
                  class="matrix-hdr-hide row-hdr-hide"
                  data-action="toggle-row"
                  data-index="${modelIndex}"
                  title="hide model"
                >x</span>
                <span class="row-hdr-idx">${rowNumber + 1}</span>
                <span class="row-hdr-name">${esc(model.display_label)}</span>
                <span class="row-hdr-score">${fmtScoreValue(score, data.meta)}</span>
              </button>
              ${cells}
              <div class="summary-cell" style="grid-row:${rowNumber + 2};">
                <span class="sum-w">${summary.wins}</span>
                <span class="sum-sep">/</span>
                <span class="sum-l">${summary.losses}</span>
                <span class="sum-sep">/</span>
                <span class="sum-n">${summary.ties}</span>
              </div>
            `;
          })
          .join("");

        return `
          <div class="matrix-wrap">
            ${renderMetaStrip(indices)}
            <div class="matrix-legend">
              <div class="legend-group">
                <span class="legend-title">row vs. column</span>
                ${legendSwatch("sig. win", "win")}
                ${legendSwatch("ns.", "ns")}
                ${legendSwatch("sig. loss", "loss")}
              </div>
              <div class="legend-group">
                <span class="legend-title">intensity</span>
                <span class="scale">
                  <span>p=α</span>
                  <span class="scale-bar"></span>
                  <span>p≤${fmtScaleP(P_INTENSITY_FLOOR)}</span>
                </span>
              </div>
              ${alphaLegend()}
              <div class="legend-group">
                <span class="legend-title">MDE80</span>
                <span class="legend-metric-value">
                  ${fmtMde(matrixMde(data.meta, state.alpha), data.meta, 1)}
                </span>
              </div>
              <div class="legend-group legend-right">
                <span class="sort-label">sort</span>
                ${sortPill("strength", "strength")}
                ${sortPill("score", "score")}
                ${sortPill("name", "name")}
                ${state.matrixSort === "anchor" && state.anchorIndex !== null ? `
                  <button
                    class="pill on anchor"
                    data-action="reset-anchor"
                  >anchored: ${esc(data.models[state.anchorIndex].display_label)} x</button>
                ` : ""}
              </div>
            </div>
            <div class="matrix-scroll">
              <div
                class="matrix-grid"
                style="
                  --cell:${cellSize}px;
                  grid-template-columns:${labelWidth}px repeat(${order.length},
                  minmax(${cellSize}px, 1fr)) ${summaryWidth}px;
                "
              >
                <div class="hdr-corner">
                  <div class="corner-y">row</div>
                  <div class="corner-x">column</div>
                  <div class="corner-diag"></div>
                </div>
                ${headers}
                <div class="summary-hdr">w / l / ns</div>
                ${rows}
              </div>
            </div>
          </div>
        `;
      }

      function renderRowsMenu(indices, visibleCount) {
        return `
          <details class="tt-dd">
            <summary class="tt-icon-btn">
              ${rowsSvg()} rows
              ${visibleCount !== indices.length
                ? `<span class="tt-pill">${visibleCount}/${indices.length}</span>`
                : ""}
            </summary>
            <div class="tt-menu">
              <div class="tt-menu-head">
                <span>models</span>
              </div>
              <div class="tt-menu-body">
                ${indices.map((index) => {
                  const model = data.models[index];
                  const checked = !state.hiddenRows.has(index) ? "checked" : "";
                  return `
                    <label class="tt-menu-row">
                      <input
                        type="checkbox"
                        data-action="toggle-row-checkbox"
                        data-index="${index}"
                        ${checked}
                      />
                      <span class="tt-menu-name">${esc(model.display_label)}</span>
                      <span class="tt-menu-n">${fmtScoreValue(displayScore(model), data.meta)}</span>
                    </label>
                  `;
                }).join("")}
              </div>
            </div>
          </details>
        `;
      }

      function renderColsMenu() {
        const visibleCount = visibleTaskColumns().length;
        return `
          <details class="tt-dd">
            <summary class="tt-icon-btn">
              ${colsSvg()} columns
              ${visibleCount !== data.task_columns.length
                ? `<span class="tt-pill">${visibleCount}/${data.task_columns.length}</span>`
                : ""}
            </summary>
            <div class="tt-menu">
              <div class="tt-menu-head">
                <span>tasks</span>
              </div>
              <div class="tt-menu-body">
                ${data.task_columns.map((column) => {
                  const checked = !state.hiddenCols.has(column.id) ? "checked" : "";
                  return `
                    <label class="tt-menu-row">
                      <input
                        type="checkbox"
                        data-action="toggle-col-checkbox"
                        data-id="${esc(column.id)}"
                        ${checked}
                      />
                      <span class="tt-menu-name">${esc(column.label)}</span>
                      <span class="tt-menu-n">task</span>
                    </label>
                  `;
                }).join("")}
              </div>
            </div>
          </details>
        `;
      }

      function renderTable(indices) {
        if (indices.length === 0) {
          return `<div class="table-wrap">${emptyState()}</div>`;
        }
        const columns = visibleTaskColumns();
        const showAverage = showAverageColumn(columns);
        const avgMeta = aggregateColumnMeta(columns);
        const sortState = resolvedTableSort(columns, showAverage);
        const rows = sortedTableRows(indices, columns, sortState);
        return `
          <div class="table-wrap">
            <div class="table-toolbar">
              <div class="tt-left">
                <span class="tt-info">
                  ${rows.length}<span class="tt-info-dim"> / ${indices.length} models</span>
                  <span class="tt-info-sep">.</span>
                  ${columns.length}
                  <span class="tt-info-dim"> / ${data.task_columns.length} tasks</span>
                </span>
              </div>
              <div class="tt-right">
                ${renderRowsMenu(indices, rows.length)}
                ${renderColsMenu()}
                <div class="tt-divider"></div>
                <button class="tt-icon-btn" data-action="export-csv">
                  ${downloadSvg()} csv
                </button>
              </div>
            </div>
            <div class="table-scroll">
              <table class="results-table">
                <thead>
                  <tr>
                    <th class="th-idx">#</th>
                    <th
                      class="th-name sortable ${sortState.key === "name" ? "active" : ""}"
                      data-action="table-sort"
                      data-key="name"
                    >
                      <span class="th-inline">
                        <span>model</span>
                        ${sortArrow("name", sortState)}
                      </span>
                    </th>
                    ${showAverage ? `
                      <th
                        class="th-avg sortable ${sortState.key === "avg" ? "active" : ""}"
                        data-action="table-sort"
                        data-key="avg"
                        title="mean across visible task columns"
                      >
                        <div class="th-stack th-sort-target">
                          <span class="th-top">avg</span>
                          <span class="th-bot th-bot-arrow">${sortArrow("avg", sortState)}</span>
                        </div>
                      </th>
                    ` : ""}
                    ${columns.map((column) => `
                      <th
                        class="th-task sortable ${sortState.key === column.id ? "active" : ""}"
                        data-action="table-sort"
                        data-key="${esc(column.id)}"
                        title="${esc(column.full_label)}"
                      >
                        <div class="th-inner">
                          <div
                            class="th-stack th-sort-target"
                          >
                            <span class="th-top">${esc(column.label)}</span>
                            <span class="th-bot th-bot-arrow">
                              ${sortArrow(column.id, sortState)}
                            </span>
                          </div>
                          <button
                            class="th-col-hide"
                            data-action="toggle-col"
                            data-id="${esc(column.id)}"
                            title="hide column"
                          >x</button>
                        </div>
                      </th>
                    `).join("")}
                  </tr>
                </thead>
                <tbody>
                  ${rows.map((model, position) => `
                    <tr>
                      <td class="td-idx">${position + 1}</td>
                      <td class="td-name">
                        <span class="td-name-text">${esc(model.display_label)}</span>
                        <button
                          class="td-name-hide"
                          data-action="toggle-row"
                          data-index="${model.index}"
                          title="hide row"
                        >x</button>
                      </td>
                      ${showAverage ? `
                        <td class="td-num td-avg">
                          ${fmtScore(averageVisibleScore(model, columns), avgMeta)}
                        </td>
                      ` : ""}
                      ${columns.map((column) => `
                        <td class="td-num">${fmtScore(model.task_scores[column.id], column)}</td>
                      `).join("")}
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>
          </div>
        `;
      }

      function renderApp() {
        const filtered = filteredModelIndices();
        root.innerHTML = `
          <div class="app">
            ${renderTopBar(filtered.indices, filtered.error)}
            <main class="main">
              ${state.view === "matrix"
                ? renderMatrix(filtered.indices)
                : renderTable(filtered.indices)}
            </main>
            <div id="pairwise-tooltip" class="tooltip hidden"></div>
          </div>
        `;
        if (hoverPair !== null) {
          hoverPair = null;
        }
      }

      function emptyState() {
        return `
          <div class="empty-state">
            <div class="empty-mark">[]</div>
            <div class="empty-title">no models match the filter</div>
            <div class="empty-sub">widen the search or add some models back in.</div>
          </div>
        `;
      }

      function legendSwatch(label, kind) {
        const config = {
          win: { bg: "oklch(0.72 0.14 150)", fg: "var(--c-paper)", mark: "+" },
          loss: { bg: "oklch(0.72 0.15 25)", fg: "var(--c-paper)", mark: "-" },
          ns: { bg: "var(--c-neutral-100)", fg: "var(--c-ink-70)", mark: "." },
        }[kind];
        return `
          <span class="legend-swatch">
            <span class="swatch" style="background:${config.bg};color:${config.fg};">
              ${config.mark}
            </span>
            <span>${esc(label)}</span>
          </span>
        `;
      }

      function sortPill(kind, label) {
        const selected =
          state.matrixSort === kind ? "pill on" : "pill";
        return `
          <button class="${selected}" data-action="matrix-sort" data-kind="${kind}">
            ${esc(label)}
          </button>
        `;
      }

      function alphaLegend() {
        return `
          <div class="legend-group legend-alpha-group">
            <span class="legend-title legend-title-alpha">α</span>
            <label class="alpha">
              <select id="alpha-select">
                ${["0.10", "0.05", "0.01", "0.001"]
                  .map((option) => {
                    const numeric = parseFloat(option);
                    const selected = numeric === state.alpha ? "selected" : "";
                    return `<option value="${option}" ${selected}>${option}</option>`;
                  })
                  .join("")}
              </select>
            </label>
          </div>
        `;
      }

      function caretSvg() {
        return `
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <path
              d="M2 3.5 L5 6.5 L8 3.5"
              stroke="currentColor"
              stroke-width="1.25"
              fill="none"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
        `;
      }

      function rowsSvg() {
        return `
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <rect x="2" y="2.5" width="12" height="2.5" rx="0.5"
              stroke="currentColor" stroke-width="1.1"/>
            <rect x="2" y="6.75" width="12" height="2.5" rx="0.5"
              stroke="currentColor" stroke-width="1.1"/>
            <rect x="2" y="11" width="12" height="2.5" rx="0.5"
              stroke="currentColor" stroke-width="1.1"/>
          </svg>
        `;
      }

      function colsSvg() {
        return `
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <rect x="2.5" y="2" width="2.5" height="12" rx="0.5"
              stroke="currentColor" stroke-width="1.1"/>
            <rect x="6.75" y="2" width="2.5" height="12" rx="0.5"
              stroke="currentColor" stroke-width="1.1"/>
            <rect x="11" y="2" width="2.5" height="12" rx="0.5"
              stroke="currentColor" stroke-width="1.1"/>
          </svg>
        `;
      }

      function downloadSvg() {
        return `
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path
              d="M8 2v8.5M8 10.5l3-3M8 10.5l-3-3"
              stroke="currentColor"
              stroke-width="1.3"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
            <path
              d="M3 12v1a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1"
              stroke="currentColor"
              stroke-width="1.3"
              stroke-linecap="round"
            />
          </svg>
        `;
      }

      function sortSvg() {
        return `
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <path class="sort-up" d="M6 2.2L8.9 5.1H3.1L6 2.2Z" />
            <path class="sort-down" d="M6 9.8L3.1 6.9H8.9L6 9.8Z" />
          </svg>
        `;
      }

      function showTooltip(row, col, event) {
        const tooltip = document.getElementById("pairwise-tooltip");
        const left = data.models[row];
        const right = data.models[col];
        const diff = data.matrix.score_diff[row]?.[col];
        const pValue = data.matrix.p_value[row]?.[col];
        const probability = data.matrix.probability[row]?.[col];
        const wins = data.matrix.wins[row]?.[col] ?? "-";
        const losses = data.matrix.losses[row]?.[col] ?? "-";
        const ties = data.matrix.ties[row]?.[col] ?? "-";
        const winRate = data.matrix.win_rate[row]?.[col];
        const se = data.matrix.se[row]?.[col];
        const direction = pairDirection(row, col);
        const deltaClass = direction > 0 ? "pos" : direction < 0 ? "neg" : "";
        const pClass = isNumber(pValue) && pValue <= state.alpha ? "sig" : "ns";
        const taskCount = data.meta.task_count ?? 0;
        const taskLabel = taskCount === 1 ? "task" : "tasks";
        const scopeLabel = data.meta.scope_kind === "suite" && taskCount > 0
          ? `${data.meta.scope_label} (${taskCount} ${taskLabel})`
          : data.meta.scope_label;
        tooltip.innerHTML = `
          <div class="tt-head">
            <div class="tt-title">paired test</div>
            <div class="tt-sub">${esc(scopeLabel)} · N=${data.meta.shared_n}</div>
          </div>
          <div class="tt-pair">
            <div class="tt-row">
              <span class="tt-dot a"></span>
              <span class="tt-name">${esc(left.display_label)}</span>
              <span class="tt-acc">${fmtScoreValue(displayScore(left), data.meta, 2)}</span>
            </div>
            <div class="tt-row">
              <span class="tt-dot b"></span>
              <span class="tt-name">${esc(right.display_label)}</span>
              <span class="tt-acc">${fmtScoreValue(displayScore(right), data.meta, 2)}</span>
            </div>
          </div>
          <div class="tt-stats">
            <div class="tt-stat">
              <span class="k">Δ (row − col)</span>
              <span class="v ${deltaClass}">${fmtDelta(diff, data.meta, 2)}</span>
            </div>
            <div class="tt-stat">
              <span class="k">p-value</span>
              <span class="v ${pClass}">
                ${fmtP(pValue)} ${isNumber(pValue) && pValue <= state.alpha
                  ? "(≤ α=" + state.alpha + ")"
                  : "(> α=" + state.alpha + ")"}
              </span>
            </div>
            <div class="tt-stat">
              <span class="k">wins / losses / ties</span>
              <span class="v mono">${wins} / ${losses} / ${ties}</span>
            </div>
          </div>
          <div class="tt-stats tt-stats-extra">
            <div class="tt-stat">
              <span class="k">win rate</span>
              <span class="v">${fmtPct(winRate, 1)}%</span>
            </div>
            <div class="tt-stat">
              <span class="k">SE</span>
              <span class="v">${fmtPct(se, 1)}%</span>
            </div>
            <div class="tt-stat">
              <span class="k">P(row > col)</span>
              <span class="v">${fmtPct(probability, 1)}%</span>
            </div>
          </div>
        `;
        tooltip.classList.remove("hidden");
        positionTooltip(event);
        highlightPair(row, col);
        hoverPair = { row, col };
      }

      function positionTooltip(event) {
        const tooltip = document.getElementById("pairwise-tooltip");
        if (!tooltip || tooltip.classList.contains("hidden")) return;
        const pad = 16;
        const width = tooltip.offsetWidth || 320;
        const height = tooltip.offsetHeight || 160;
        let x = event.clientX + pad;
        let y = event.clientY + pad;
        if (x + width + pad > window.innerWidth) x = event.clientX - width - pad;
        if (y + height + pad > window.innerHeight) y = event.clientY - height - pad;
        tooltip.style.left = x + "px";
        tooltip.style.top = y + "px";
      }

      function clearHighlight() {
        root.querySelectorAll(".hover").forEach((node) => node.classList.remove("hover"));
        root.querySelectorAll(".axis-hi").forEach((node) => node.classList.remove("axis-hi"));
      }

      function highlightPair(row, col) {
        clearHighlight();
        root.querySelectorAll(".cell[data-row]").forEach((node) => {
          const nodeRow = parseInt(node.dataset.row, 10);
          const nodeCol = parseInt(node.dataset.col, 10);
          if (nodeRow === row && nodeCol === col) {
            node.classList.add("hover");
          }
          if (nodeRow === row || nodeRow === col || nodeCol === row || nodeCol === col) {
            node.classList.add("axis-hi");
          }
        });
        root
          .querySelectorAll(".row-hdr[data-row-index], .col-hdr[data-col-index]")
          .forEach((node) => {
            const nodeRow = node.dataset.rowIndex ? parseInt(node.dataset.rowIndex, 10) : null;
            const nodeCol = node.dataset.colIndex ? parseInt(node.dataset.colIndex, 10) : null;
            if (nodeRow === row || nodeRow === col || nodeCol === row || nodeCol === col) {
              node.classList.add("axis-hi");
            }
          });
      }

      function hideTooltip() {
        const tooltip = document.getElementById("pairwise-tooltip");
        if (tooltip) tooltip.classList.add("hidden");
        clearHighlight();
        hoverPair = null;
      }

      function exportCsv() {
        const filtered = filteredModelIndices().indices;
        const columns = visibleTaskColumns();
        const showAverage = showAverageColumn(columns);
        const avgMeta = aggregateColumnMeta(columns);
        const sortState = resolvedTableSort(columns, showAverage);
        const rows = sortedTableRows(filtered, columns, sortState);
        const header = showAverage
          ? ["model", "avg", ...columns.map((column) => column.full_label)]
          : ["model", ...columns.map((column) => column.full_label)];
        const body = rows.map((model) => showAverage
          ? [
              model.display_label,
              fmtScore(averageVisibleScore(model, columns), avgMeta, 2),
              ...columns.map((column) => fmtScore(model.task_scores[column.id], column, 2)),
            ]
          : [
              model.display_label,
              ...columns.map((column) => fmtScore(model.task_scores[column.id], column, 2)),
            ]);
        const lines = [header, ...body].map((row) =>
          row
            .map((value) => {
              const text = String(value ?? "");
              return /[,"\\n]/.test(text)
                ? '"' + text.replace(/"/g, '""') + '"'
                : text;
            })
            .join(",")
        );
        const blob = new Blob([lines.join("\\n") + "\\n"], {
          type: "text/csv;charset=utf-8",
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = data.meta.storage_key + ".csv";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      }

      root.addEventListener("click", (event) => {
        const target = event.target.closest("[data-action]");
        if (!target) return;
        const action = target.dataset.action;
        if (action === "view") {
          state.view = target.dataset.view;
          persistState();
          renderApp();
          return;
        }
        if (action === "matrix-sort") {
          state.matrixSort = target.dataset.kind;
          if (state.matrixSort !== "anchor") state.anchorIndex = null;
          renderApp();
          return;
        }
        if (action === "anchor") {
          state.matrixSort = "anchor";
          state.anchorIndex = parseInt(target.dataset.index, 10);
          renderApp();
          return;
        }
        if (action === "reset-anchor") {
          state.matrixSort = "strength";
          state.anchorIndex = null;
          renderApp();
          return;
        }
        if (action === "toggle-row") {
          const index = parseInt(target.dataset.index, 10);
          if (state.hiddenRows.has(index)) state.hiddenRows.delete(index);
          else state.hiddenRows.add(index);
          renderApp();
          return;
        }
        if (action === "toggle-col") {
          const id = target.dataset.id;
          if (state.hiddenCols.has(id)) state.hiddenCols.delete(id);
          else state.hiddenCols.add(id);
          renderApp();
          return;
        }
        if (action === "table-sort") {
          const key = target.dataset.key;
          if (state.tableSortKey === key) {
            state.tableSortDir = state.tableSortDir === "asc" ? "desc" : "asc";
          } else {
            state.tableSortKey = key;
            if (key === "name") {
              state.tableSortDir = "asc";
            } else {
              const visibleColumns = visibleTaskColumns();
              if (key === "avg") {
                state.tableSortDir = defaultScoreSortDir(aggregateColumnMeta(visibleColumns));
              } else {
                const column = visibleColumns.find((entry) => entry.id === key);
                state.tableSortDir = defaultScoreSortDir(column);
              }
            }
          }
          renderApp();
          return;
        }
        if (action === "export-csv") {
          exportCsv();
        }
      });

      root.addEventListener("change", (event) => {
        const target = event.target;
        if (target.id === "alpha-select") {
          state.alpha = parseFloat(target.value);
          persistState();
          renderApp();
          return;
        }
        const action = target.dataset.action;
        if (action === "toggle-row-checkbox") {
          const index = parseInt(target.dataset.index, 10);
          if (target.checked) state.hiddenRows.delete(index);
          else state.hiddenRows.add(index);
          renderApp();
          return;
        }
        if (action === "toggle-col-checkbox") {
          const id = target.dataset.id;
          if (target.checked) state.hiddenCols.delete(id);
          else state.hiddenCols.add(id);
          renderApp();
        }
      });

      root.addEventListener("input", (event) => {
        const target = event.target;
        if (target.id === "regex-filter") {
          state.regex = target.value;
          persistState();
          renderApp();
        }
      });

      root.addEventListener("mouseover", (event) => {
        const cell = event.target.closest(".cell[data-row][data-col]");
        if (!cell || !root.contains(cell)) return;
        const row = parseInt(cell.dataset.row, 10);
        const col = parseInt(cell.dataset.col, 10);
        showTooltip(row, col, event);
      });

      root.addEventListener("mousemove", (event) => {
        if (hoverPair !== null) {
          positionTooltip(event);
        }
      });

      root.addEventListener("mouseout", (event) => {
        const cell = event.target.closest(".cell[data-row][data-col]");
        if (!cell) return;
        const related = event.relatedTarget;
        if (related && cell.contains(related)) return;
        hideTooltip();
      });

      renderApp();
    })();
