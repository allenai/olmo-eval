(() => {
      const pageData = window.RESULTS_VIEWER_DATA;
      const root = document.getElementById("browser-root");
      const scopeForm = document.querySelector(".scope-form");
      const modelFilterDetails = document.getElementById("model-filter-details");
      const storageBase = "pairwise-browser:" +
        (pageData.group_data?.summary?.group_name || "root") +
        ":";
      const P_INTENSITY_FLOOR = 1e-12;
      const state = {
        view: loadState("view", "matrix"),
        regex: loadState("regex", ""),
        alpha: 0.05,
        matrixSort: "strength",
        anchorIndex: null,
        tableSortKey: "avg",
        tableSortDir: "desc",
        excludedModels: loadSetState("excludedModels"),
        hiddenCols: new Set(),
        columnsMenuOpen: false,
      };
      let hoverPair = null;
      let pendingNavigation = false;
      let activeTableScrollDrag = null;
      const resultsScrollbarObservers = new WeakMap();
      let resultsScrollSyncQueued = false;
      let searchSelectWidthSyncQueued = false;
      const textMeasureCanvas = document.createElement("canvas");
      const textMeasureContext = textMeasureCanvas.getContext("2d");

      function loadState(key, fallback) {
        const value = window.localStorage.getItem(storageBase + key);
        return value === null ? fallback : value;
      }

      function loadSetState(key) {
        try {
          const value = JSON.parse(loadState(key, "[]"));
          return new Set(Array.isArray(value) ? value.map((item) => String(item)) : []);
        } catch (_error) {
          return new Set();
        }
      }

      function persistState() {
        window.localStorage.setItem(storageBase + "view", state.view);
        window.localStorage.setItem(storageBase + "regex", state.regex);
        window.localStorage.setItem(
          storageBase + "excludedModels",
          JSON.stringify(Array.from(state.excludedModels))
        );
      }

      function clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
      }

      function measuredTextWidth(text, font) {
        if (!textMeasureContext) return 0;
        textMeasureContext.font = font;
        return textMeasureContext.measureText(String(text || "")).width;
      }

      function nodeFont(node) {
        const style = window.getComputedStyle(node);
        return style.font || [
          style.fontStyle,
          style.fontVariant,
          style.fontWeight,
          style.fontSize,
          style.fontFamily,
        ].filter(Boolean).join(" ");
      }

      function measureNodeTextWidth(node, text = null) {
        if (!node) return 0;
        return measuredTextWidth(text ?? node.textContent ?? "", nodeFont(node));
      }

      function syncSearchSelectMenuWidth(control) {
        if (!control) return;
        if (!control.classList.contains("group-select") && !control.classList.contains("scope-select")) {
          return;
        }
        const summary = control.querySelector(".search-select-summary");
        const summaryText = control.querySelector(".control-summary-text");
        const filterInput = control.querySelector(".search-select-filter");
        const optionLabels = Array.from(control.querySelectorAll(".search-select-option-main"));
        let widestText = Math.max(
          measureNodeTextWidth(summaryText),
          measureNodeTextWidth(filterInput, filterInput?.placeholder || "")
        );
        optionLabels.forEach((node) => {
          widestText = Math.max(widestText, measureNodeTextWidth(node));
        });
        const summaryWidth = Math.ceil(summary?.getBoundingClientRect().width || 0);
        const desiredWidth = Math.ceil(widestText + 44);
        const viewportMax = Math.max(320, window.innerWidth - 48);
        const width = clamp(desiredWidth, summaryWidth, viewportMax);
        control.style.setProperty("--search-select-menu-width", `${width}px`);
      }

      function syncAllSearchSelectMenuWidths() {
        scopeForm?.querySelectorAll("[data-search-select]").forEach((control) => {
          syncSearchSelectMenuWidth(control);
        });
      }

      function queueSearchSelectMenuWidthSync() {
        if (searchSelectWidthSyncQueued) return;
        searchSelectWidthSyncQueued = true;
        window.requestAnimationFrame(() => {
          searchSelectWidthSyncQueued = false;
          syncAllSearchSelectMenuWidths();
        });
      }

      function queueResultsScrollSync() {
        if (resultsScrollSyncQueued) return;
        resultsScrollSyncQueued = true;
        window.requestAnimationFrame(() => {
          resultsScrollSyncQueued = false;
          syncResultsScrollbars();
        });
      }

      function showScopeLoading() {
        pendingNavigation = true;
        document.body.classList.add("is-page-loading");
        if (root) root.setAttribute("aria-busy", "true");
        if (scopeForm) scopeForm.classList.add("is-loading");
      }

      function closeSearchSelects(except = null) {
        document.querySelectorAll(".search-select-dd[open]").forEach((details) => {
          if (details !== except) details.open = false;
        });
      }

      function buildScopeUrl(form) {
        const url = new URL(window.location.href);
        const params = new URLSearchParams();
        Array.from(new FormData(form).entries()).forEach(([key, value]) => {
          const text = String(value ?? "").trim();
          if (!text) return;
          params.append(key, text);
        });
        url.search = params.toString();
        return url.toString();
      }

      function submitScopeForm() {
        if (!scopeForm || pendingNavigation) return;
        const url = buildScopeUrl(scopeForm);
        closeSearchSelects();
        showScopeLoading();
        scopeForm.querySelectorAll("details[open]").forEach((details) => {
          details.open = false;
        });
        scopeForm.querySelectorAll("select, input, button").forEach((control) => {
          if (control.tagName === "INPUT" && control.type === "hidden") return;
          control.disabled = true;
        });
        scopeForm.querySelectorAll("summary").forEach((summary) => {
          summary.setAttribute("aria-disabled", "true");
        });
        window.setTimeout(() => {
          window.location.assign(url);
        }, 40);
      }

      function filterSearchSelect(control, query) {
        const needle = String(query || "").trim().toLowerCase();
        let visible = 0;
        searchSelectOptions(control).forEach((option) => {
          const haystack = String(
            option.dataset.filterText ||
            option.dataset.summaryText ||
            option.textContent ||
            ""
          ).toLowerCase();
          const show = !needle || haystack.includes(needle);
          option.hidden = !show;
          if (!show) {
            option.dataset.matchRank = "";
            option.style.order = String(searchOptionIndex(option));
            return;
          }
          const matchRank = searchOptionScore(option, needle);
          option.dataset.matchRank = needle ? String(matchRank) : "";
          option.style.order = String((needle ? matchRank * 1000 : 0) + searchOptionIndex(option));
          visible += 1;
        });
        const empty = control.querySelector('[data-role="search-select-empty"]');
        if (empty) empty.hidden = visible > 0;
        setActiveSearchOption(control, bestVisibleSearchOption(control, needle), {
          scroll: Boolean(needle),
        });
      }

      function searchSelectOptions(control) {
        return Array.from(control.querySelectorAll('[data-role="search-select-option"]'));
      }

      function searchOptionIndex(option) {
        const index = parseInt(option?.dataset.optionIndex || "", 10);
        return Number.isFinite(index) ? index : 0;
      }

      function searchOptionScore(option, query) {
        const needle = String(query || "").trim().toLowerCase();
        if (!needle) return option.classList.contains("is-selected") ? -1 : 9;
        const texts = new Set([
          option.dataset.summaryText || "",
          option.dataset.filterText || "",
          option.dataset.value || "",
          option.textContent || "",
        ]);
        let score = 9;
        texts.forEach((value) => {
          const text = String(value || "").trim().toLowerCase();
          if (!text) return;
          if (text === needle) {
            score = Math.min(score, 0);
            return;
          }
          if (text.startsWith(needle)) {
            score = Math.min(score, 1);
            return;
          }
          if (text.split(/[\\s:/()_.-]+/).some((part) => part.startsWith(needle))) {
            score = Math.min(score, 2);
            return;
          }
          if (text.includes(needle)) {
            score = Math.min(score, 3);
          }
        });
        return score;
      }

      function orderedVisibleSearchOptions(control) {
        return searchSelectOptions(control)
          .filter((option) => !option.hidden)
          .sort((left, right) => {
            const leftRank = parseInt(left.dataset.matchRank || "", 10);
            const rightRank = parseInt(right.dataset.matchRank || "", 10);
            const resolvedLeftRank = Number.isFinite(leftRank) ? leftRank : 0;
            const resolvedRightRank = Number.isFinite(rightRank) ? rightRank : 0;
            if (resolvedLeftRank !== resolvedRightRank) {
              return resolvedLeftRank - resolvedRightRank;
            }
            return searchOptionIndex(left) - searchOptionIndex(right);
          });
      }

      function activeSearchOption(control) {
        return control.querySelector('[data-role="search-select-option"].is-active:not([hidden])');
      }

      function setActiveSearchOption(control, option, { scroll = false } = {}) {
        if (!control) return;
        searchSelectOptions(control).forEach((node) => {
          node.classList.toggle("is-active", node === option && !node.hidden);
        });
        if (scroll && option) {
          option.scrollIntoView({ block: "nearest" });
        }
      }

      function bestVisibleSearchOption(control, query) {
        const visible = orderedVisibleSearchOptions(control);
        if (!visible.length) return null;
        const needle = String(query || "").trim();
        if (!needle) {
          return visible.find((option) => option.classList.contains("is-selected")) || visible[0];
        }
        return visible[0];
      }

      function resetSearchSelect(control, focus = false) {
        if (!control) return;
        const filterInput = control.querySelector('[data-role="search-select-filter"]');
        if (filterInput) filterInput.value = "";
        filterSearchSelect(control, "");
        if (focus && filterInput) {
          window.setTimeout(() => {
            filterInput.focus();
            filterInput.select?.();
          }, 0);
        }
      }

      function moveActiveSearchOption(control, step) {
        const visible = orderedVisibleSearchOptions(control);
        if (!visible.length) return null;
        const current = activeSearchOption(control);
        const currentIndex = current ? visible.indexOf(current) : -1;
        const nextIndex = currentIndex < 0
          ? (step > 0 ? 0 : visible.length - 1)
          : Math.max(0, Math.min(visible.length - 1, currentIndex + step));
        const option = visible[nextIndex];
        setActiveSearchOption(control, option, { scroll: true });
        return option;
      }

      function setSearchSelectSummary(control, option) {
        if (!control || !option) return;
        const summary = control.querySelector(".search-select-summary");
        const summaryText = control.querySelector(".control-summary-text");
        const nextText = String(
          option.dataset.summaryText || option.dataset.filterText || option.textContent || ""
        ).trim();
        if (summaryText) summaryText.textContent = nextText;
        if (summary) summary.title = nextText;
        searchSelectOptions(control).forEach((node) => {
          node.classList.toggle("is-selected", node === option);
        });
        setActiveSearchOption(control, option);
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

      function modelKey(model) {
        if (model?.timestamp) return modelExportRef(model);
        return String(
          model?.model_hash ||
          model?.model_name ||
          model?.display_label ||
          model?.index ||
          ""
        );
      }

      function allFilterModels() {
        const resultsModels = pageData.group_data?.results_table?.models;
        if (Array.isArray(resultsModels) && resultsModels.length > 0) {
          return resultsModels;
        }
        const pairwiseModels = pageData.pairwise_data?.models;
        return Array.isArray(pairwiseModels) ? pairwiseModels : [];
      }

      function trimExcludedModels() {
        const validKeys = new Set(allFilterModels().map((model) => modelKey(model)));
        let changed = false;
        state.excludedModels.forEach((key) => {
          if (!validKeys.has(key)) {
            state.excludedModels.delete(key);
            changed = true;
          }
        });
        if (changed) persistState();
      }

      function isExcludedModel(model) {
        return state.excludedModels.has(modelKey(model));
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

      function compileRegex() {
        if (!state.regex) return { regex: null, error: false };
        try {
          return { regex: new RegExp(state.regex, "i"), error: false };
        } catch (_error) {
          return { regex: null, error: true };
        }
      }

      function filteredResultsModelIndices(resultsData) {
        const compiled = compileRegex();
        return {
          indices: resultsData.models
            .filter((model) => {
              if (isExcludedModel(model)) return false;
              if (compiled.error || !compiled.regex) return true;
              return compiled.regex.test(model.display_label);
            })
            .map((model) => model.index),
          error: compiled.error,
        };
      }

      function filteredPairwiseModelIndices(pairwiseData) {
        if (!pairwiseData) {
          return { indices: [], error: compileRegex().error };
        }
        const compiled = compileRegex();
        return {
          indices: pairwiseData.models
            .filter((model) => {
              if (isExcludedModel(model)) return false;
              if (compiled.error || !compiled.regex) return true;
              return compiled.regex.test(model.display_label);
            })
            .map((model) => model.index),
          error: compiled.error,
        };
      }

      function displayScore(model) {
        if (isNumber(model.display_score)) return model.display_score;
        if (isNumber(model.avg_task_score)) return model.avg_task_score;
        if (isNumber(model.avg_score)) return model.avg_score;
        return null;
      }

      function comparisonDiff(pairwiseData, row, col) {
        const diff = pairwiseData.matrix.score_diff[row]?.[col];
        return comparisonValue(diff, pairwiseData.meta);
      }

      function pairDirection(pairwiseData, row, col) {
        const diff = comparisonDiff(pairwiseData, row, col);
        if (isNumber(diff) && diff !== 0) return Math.sign(diff);
        const winRate = pairwiseData.matrix.win_rate[row]?.[col];
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

      function matrixOrder(pairwiseData, indices) {
        const ordered = indices.slice();
        if (state.matrixSort === "name") {
          ordered.sort((a, b) => {
            const left = pairwiseData.models[a].display_label;
            const right = pairwiseData.models[b].display_label;
            return left.localeCompare(right);
          });
          return ordered;
        }
        if (state.matrixSort === "score") {
          ordered.sort((a, b) => {
            const av = comparisonValue(displayScore(pairwiseData.models[a]), pairwiseData.meta);
            const bv = comparisonValue(displayScore(pairwiseData.models[b]), pairwiseData.meta);
            return (bv ?? -Infinity) - (av ?? -Infinity);
          });
          return ordered;
        }
        if (state.matrixSort === "anchor" && indices.includes(state.anchorIndex)) {
          const anchor = state.anchorIndex;
          ordered.sort((a, b) => {
            if (a === anchor) return -1;
            if (b === anchor) return 1;
            const av = comparisonDiff(pairwiseData, a, anchor);
            const bv = comparisonDiff(pairwiseData, b, anchor);
            const da = isNumber(av) ? av : pairwiseData.matrix.win_rate[a]?.[anchor] ?? 0.5;
            const db = isNumber(bv) ? bv : pairwiseData.matrix.win_rate[b]?.[anchor] ?? 0.5;
            return db - da;
          });
          return ordered;
        }
        ordered.sort((a, b) => {
          const av = pairwiseData.models[a].strength ?? -Infinity;
          const bv = pairwiseData.models[b].strength ?? -Infinity;
          if (bv !== av) return bv - av;
          return (
            comparisonValue(displayScore(pairwiseData.models[b]), pairwiseData.meta) ?? -Infinity
          ) - (
            comparisonValue(displayScore(pairwiseData.models[a]), pairwiseData.meta) ?? -Infinity
          );
        });
        return ordered;
      }

      function summaryFor(pairwiseData, modelIndex, indices) {
        let wins = 0;
        let losses = 0;
        let ties = 0;
        indices.forEach((other) => {
          if (other === modelIndex) return;
          const pValue = pairwiseData.matrix.p_value[modelIndex]?.[other];
          const direction = pairDirection(pairwiseData, modelIndex, other);
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
        return indices;
      }

      function selectedScopeOption() {
        const scopeOptions = pageData.group_data?.scope_options;
        const selectedScopeKey = pageData.selected_scope_key;
        if (!Array.isArray(scopeOptions) || !selectedScopeKey) return null;
        return scopeOptions.find((option) => option.key === selectedScopeKey) || null;
      }

      function scopedTaskColumns(resultsData) {
        const scopeOption = selectedScopeOption();
        if (
          !scopeOption ||
          !Array.isArray(scopeOption.task_ids) ||
          scopeOption.task_ids.length === 0
        ) {
          return resultsData.task_columns;
        }
        const allowedTaskIds = new Set(scopeOption.task_ids.map((taskId) => String(taskId)));
        return resultsData.task_columns.filter((column) => allowedTaskIds.has(column.id));
      }

      function visibleTaskColumns(resultsData) {
        return scopedTaskColumns(resultsData).filter((column) => !state.hiddenCols.has(column.id));
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

      function sortedTableRows(resultsData, indices, columns, sortState) {
        const rowLookup = new Map(resultsData.models.map((model) => [model.index, model]));
        const rows = visibleRows(indices).map((index) => rowLookup.get(index));
        rows.sort((left, right) => {
          if (sortState.key === "name") {
            return compareValues(left.display_label, right.display_label, sortState.dir);
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

      function renderResults(groupData) {
        const resultsData = groupData?.results_table;
        if (!pageData.has_groups) {
          return `
            <div class="browser-section">
              <div class="empty-state">
                <div class="empty-mark">[]</div>
                <div class="empty-title">no experiment groups found</div>
                <div class="empty-sub">
                  connect to a populated results database to start exploring.
                </div>
              </div>
            </div>
          `;
        }
        if (!pageData.selected_group) {
          return `
            <div class="browser-section">
              <div class="empty-state">
                <div class="empty-mark">[]</div>
                <div class="empty-title">pick an experiment group and suite or task</div>
                <div class="empty-sub">
                  use the selectors above to choose what you want to compare.
                </div>
              </div>
            </div>
          `;
        }
        if (!resultsData) {
          return `
            <div class="table-wrap">
              <div class="empty-state">
                <div class="empty-mark">[]</div>
                <div class="empty-title">no task results found</div>
                <div class="empty-sub">
                  choose another group or run evaluations for this one first.
                </div>
              </div>
            </div>
          `;
        }
        const filtered = filteredResultsModelIndices(resultsData);
        if (filtered.indices.length === 0) {
          return `
            <div class="table-wrap">
              ${emptyState()}
            </div>
          `;
        }
        const scopedColumns = scopedTaskColumns(resultsData);
        const columns = visibleTaskColumns(resultsData);
        const showAverage = showAverageColumn(columns);
        const avgMeta = aggregateColumnMeta(columns);
        const sortState = resolvedTableSort(columns, showAverage);
        const rows = sortedTableRows(resultsData, filtered.indices, columns, sortState);
        return `
          <div class="table-wrap">
            <div class="table-toolbar">
              <div class="tt-left">
                <span class="tt-info">
                  ${filtered.indices.length}
                  <span class="tt-info-dim">
                    / ${resultsData.models.length} models
                  </span>
                  <span class="tt-info-sep">.</span>
                  ${columns.length}
                  <span class="tt-info-dim">
                    / ${scopedColumns.length} tasks
                  </span>
                </span>
              </div>
              <div class="tt-right">
                ${renderColsMenu(resultsData)}
                <div class="tt-divider"></div>
                <button class="tt-icon-btn" data-action="export-csv">
                  ${downloadSvg()} csv
                </button>
              </div>
            </div>
            <div class="table-scroll-shell">
              <div class="table-scroll" data-role="results-scroll-region">
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
                        <button
                          class="td-name-hide"
                          data-action="exclude-model"
                          data-model-key="${esc(modelKey(model))}"
                          title="exclude model"
                        >x</button>
                        <span class="td-name-main">
                          <span class="td-name-text">${esc(model.display_label)}</span>
                        </span>
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
              <div class="table-xbar" data-role="results-scrollbar" hidden>
                <span class="table-xbar-label">scroll</span>
                <div class="table-xbar-track" data-role="results-scrollbar-track">
                  <div class="table-xbar-thumb" data-role="results-scrollbar-thumb"></div>
                </div>
              </div>
            </div>
          </div>
        `;
      }

      function renderColsMenu(resultsData) {
        const scopedColumns = scopedTaskColumns(resultsData);
        const visibleCount = visibleTaskColumns(resultsData).length;
        const hiddenCount = Math.max(0, scopedColumns.length - visibleCount);
        return `
          <details class="tt-dd tt-cols-menu" ${state.columnsMenuOpen ? "open" : ""}>
            <summary class="tt-icon-btn">
              ${colsSvg()} columns
              ${visibleCount !== scopedColumns.length
                ? `<span class="tt-pill">${visibleCount}/${scopedColumns.length}</span>`
                : ""}
            </summary>
            <div class="tt-menu">
              <div class="tt-menu-head">
                <span>tasks</span>
                ${hiddenCount > 0 ? `
                  <button
                    type="button"
                    class="tt-menu-clear"
                    data-action="reset-cols"
                    title="show all columns"
                  >reset</button>
                ` : ""}
              </div>
              <div class="tt-menu-body">
                ${scopedColumns.map((column) => {
                  const checked = !state.hiddenCols.has(column.id) ? "checked" : "";
                  return `
                    <div class="tt-menu-row">
                      <input
                        type="checkbox"
                        data-action="toggle-col-checkbox"
                        data-id="${esc(column.id)}"
                        ${checked}
                      />
                      <button
                        type="button"
                        class="tt-menu-name-btn"
                        data-action="solo-col"
                        data-id="${esc(column.id)}"
                        title="show only this column"
                      >
                        <span class="tt-menu-name">${esc(column.label)}</span>
                      </button>
                      <span class="tt-menu-n">${column.model_count}</span>
                    </div>
                  `;
                }).join("")}
              </div>
            </div>
          </details>
        `;
      }

      function bindColumnsMenu() {
        const colsMenu = root.querySelector(".tt-cols-menu");
        if (!colsMenu || colsMenu.dataset.bound === "1") return;
        colsMenu.dataset.bound = "1";
        colsMenu.addEventListener("toggle", () => {
          state.columnsMenuOpen = colsMenu.open;
        });
      }

      function bindResultsScrollbars() {
        root.querySelectorAll(".table-scroll-shell").forEach((shell) => {
          if (shell.dataset.scrollbarBound === "1") return;
          const region = shell.querySelector('[data-role="results-scroll-region"]');
          const track = shell.querySelector('[data-role="results-scrollbar-track"]');
          const thumb = shell.querySelector('[data-role="results-scrollbar-thumb"]');
          if (!region || !track || !thumb) return;
          shell.dataset.scrollbarBound = "1";

          region.addEventListener("scroll", syncResultsScrollbars);

          const table = region.querySelector(".results-table");
          if (table && "ResizeObserver" in window && !resultsScrollbarObservers.has(shell)) {
            const observer = new ResizeObserver(() => {
              queueResultsScrollSync();
            });
            observer.observe(region);
            observer.observe(table);
            resultsScrollbarObservers.set(shell, observer);
          }

          track.addEventListener("pointerdown", (event) => {
            if (event.button !== 0) return;
            const metrics = measureResultsScrollMetrics(region);
            const rect = track.getBoundingClientRect();
            const thumbWidth = thumb.offsetWidth || 0;
            const maxOffset = Math.max(0, rect.width - thumbWidth);
            const maxScroll = metrics.maxScroll;
            if (maxOffset <= 0 || maxScroll <= 0) return;
            const nextOffset = clamp(
              event.clientX - rect.left - thumbWidth / 2,
              0,
              maxOffset
            );
            region.scrollLeft = (nextOffset / maxOffset) * maxScroll;
          });

          thumb.addEventListener("pointerdown", (event) => {
            if (event.button !== 0) return;
            event.preventDefault();
            event.stopPropagation();
            const metrics = measureResultsScrollMetrics(region);
            const trackWidth = track.clientWidth;
            const thumbWidth = thumb.offsetWidth || 0;
            const maxOffset = Math.max(0, trackWidth - thumbWidth);
            const maxScroll = metrics.maxScroll;
            if (maxOffset <= 0 || maxScroll <= 0) return;
            activeTableScrollDrag = {
              region,
              maxOffset,
              maxScroll,
              startX: event.clientX,
              startScrollLeft: region.scrollLeft,
            };
            document.body.classList.add("is-dragging-table-xbar");
          });
        });
      }

      function measureResultsScrollMetrics(region) {
        const table = region?.querySelector(".results-table");
        if (!region || !table) {
          return {
            table,
            regionWidth: 0,
            contentWidth: 0,
            maxScroll: 0,
          };
        }
        const regionWidth =
          region.clientWidth ||
          Math.round(region.getBoundingClientRect().width) ||
          0;
        const actualScrollWidth = region.scrollWidth || 0;
        const fallbackWidth =
          table.scrollWidth ||
          table.offsetWidth ||
          Math.round(table.getBoundingClientRect().width) ||
          0;
        const contentWidth = actualScrollWidth || fallbackWidth;
        const maxScroll =
          actualScrollWidth > 0
            ? Math.max(0, actualScrollWidth - regionWidth)
            : Math.max(0, fallbackWidth - regionWidth);
        return {
          table,
          regionWidth,
          contentWidth,
          maxScroll,
        };
      }

      function syncResultsScrollbars() {
        root.querySelectorAll(".table-scroll-shell").forEach((shell) => {
          const region = shell.querySelector('[data-role="results-scroll-region"]');
          const rail = shell.querySelector('[data-role="results-scrollbar"]');
          const track = shell.querySelector('[data-role="results-scrollbar-track"]');
          const thumb = shell.querySelector('[data-role="results-scrollbar-thumb"]');
          const { table, regionWidth, contentWidth, maxScroll } = measureResultsScrollMetrics(region);
          if (!region || !rail || !track || !thumb || !table) return;

          if (maxScroll <= 1) {
            rail.hidden = true;
            shell.classList.remove("has-x-overflow");
            region.scrollLeft = 0;
            thumb.style.width = "";
            thumb.style.transform = "";
            return;
          }

          rail.hidden = false;
          shell.classList.add("has-x-overflow");
          const trackWidth =
            track.clientWidth ||
            Math.round(track.getBoundingClientRect().width) ||
            0;
          if (trackWidth <= 0) {
            queueResultsScrollSync();
            return;
          }
          const thumbWidth = Math.max(
            56,
            Math.round(trackWidth * (regionWidth / contentWidth))
          );
          const maxOffset = Math.max(0, trackWidth - thumbWidth);
          const scrollLeft = clamp(region.scrollLeft, 0, maxScroll);
          const offset = maxScroll <= 0
            ? 0
            : (scrollLeft / maxScroll) * maxOffset;
          thumb.style.width = `${thumbWidth}px`;
          thumb.style.transform = `translateX(${offset}px)`;
        });
      }

      function renderMatrix(pairwiseData, errorMessage, errorDetails) {
        if (!pageData.has_groups) {
          return `
            <div class="matrix-wrap">
              <div class="empty-state">
                <div class="empty-mark">[]</div>
                <div class="empty-title">no experiment groups found</div>
                <div class="empty-sub">
                  connect to a populated results database to start exploring.
                </div>
              </div>
            </div>
          `;
        }
        if (!pageData.selected_group) {
          return `
            <div class="matrix-wrap">
              <div class="empty-state">
                <div class="empty-mark">[]</div>
                <div class="empty-title">pick an experiment group and suite or task</div>
                <div class="empty-sub">
                  use the selectors above to choose what you want to compare.
                </div>
              </div>
            </div>
          `;
        }
        if (!pageData.selected_scope_key) {
          return `
            <div class="matrix-wrap">
              <div class="single-model-note">
                <div>pick a suite or task to open the paired-test view.</div>
              </div>
            </div>
          `;
        }
        if (errorMessage || errorDetails) {
          return renderPairwiseError(errorDetails, errorMessage, pageData.group_data);
        }
        if (!pairwiseData) {
          return `
            <div class="matrix-wrap">
              <div class="single-model-note">
                <div>pick a suite or task to open the paired-test view.</div>
              </div>
            </div>
          `;
        }
        const filtered = filteredPairwiseModelIndices(pairwiseData);
        const indices = filtered.indices;
        if (indices.length === 0) {
          return `
            <div class="matrix-wrap">
              ${emptyState()}
            </div>
          `;
        }
        if (indices.length === 1) {
          return `
            <div class="matrix-wrap">
              <div class="single-model-note">
                <div>only one model is currently visible.</div>
                <div class="dim">show at least two models to run a paired test.</div>
              </div>
            </div>
          `;
        }
        const order = matrixOrder(pairwiseData, indices);
        const cellSize = 40;
        const labelWidth = 240;
        const summaryWidth = 112;
        const gridColumns = [
          `${labelWidth}px`,
          `repeat(${order.length}, minmax(${cellSize}px, 1fr))`,
          `${summaryWidth}px`,
        ].join(" ");
        return `
          <div class="matrix-wrap">
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
                  ${fmtMde(matrixMde(pairwiseData.meta, state.alpha), pairwiseData.meta, 1)}
                </span>
              </div>
              <div class="legend-group legend-right">
                <span class="sort-label">sort</span>
                ${sortPill("strength", "strength")}
                ${sortPill("score", "score")}
                ${sortPill("name", "name")}
                ${state.matrixSort === "anchor" &&
                  state.anchorIndex !== null &&
                  indices.includes(state.anchorIndex) ? `
                  <button class="pill on anchor" data-action="reset-anchor">
                    anchored: ${esc(pairwiseData.models[state.anchorIndex].display_label)} x
                  </button>
                ` : ""}
              </div>
              ${renderPairwiseExportMenu()}
            </div>
            <div class="matrix-scroll">
              <div
                class="matrix-grid"
                style="
                  --cell:${cellSize}px;
                  grid-template-columns:${gridColumns};
                "
              >
                <div class="hdr-corner">
                  <div class="corner-y">row</div>
                  <div class="corner-x">column</div>
                  <div class="corner-diag"></div>
                </div>
                ${order.map((modelIndex, position) => `
                  <button
                    class="col-hdr ${state.anchorIndex === modelIndex ? "anchored" : ""}"
                    style="grid-column:${position + 2};"
                    data-action="anchor"
                    data-index="${modelIndex}"
                    data-col-index="${modelIndex}"
                    title="anchor on ${esc(pairwiseData.models[modelIndex].display_label)}"
                  >
                    <span
                      class="matrix-hdr-hide col-hdr-hide"
                      data-action="exclude-model"
                      data-model-key="${esc(modelKey(pairwiseData.models[modelIndex]))}"
                      title="exclude model"
                    >x</span>
                    <span class="col-hdr-inner">
                      <span class="col-hdr-name">
                        ${esc(pairwiseData.models[modelIndex].display_label)}
                      </span>
                    </span>
                  </button>
                `).join("")}
                <div class="summary-hdr">w / l / ns</div>
                ${order.map((modelIndex, rowNumber) => {
                  const model = pairwiseData.models[modelIndex];
                  const summary = summaryFor(pairwiseData, modelIndex, order);
                  return `
                    <button
                      class="row-hdr ${state.anchorIndex === modelIndex ? "anchored" : ""}"
                      style="grid-row:${rowNumber + 2};"
                      data-action="anchor"
                      data-index="${modelIndex}"
                      data-row-index="${modelIndex}"
                      title="anchor on ${esc(model.display_label)}"
                    >
                      <span class="row-hdr-main">
                        <span class="row-hdr-idx">${rowNumber + 1}</span>
                        <span
                          class="matrix-hdr-hide row-hdr-hide"
                          data-action="exclude-model"
                          data-model-key="${esc(modelKey(model))}"
                          title="exclude model"
                        >x</span>
                        <span class="row-hdr-name">${esc(model.display_label)}</span>
                      </span>
                      <span class="row-hdr-score">${fmtScoreValue(displayScore(model), pairwiseData.meta)}</span>
                    </button>
                    ${order.map((otherIndex, colNumber) => {
                      const rowStyle = `grid-row:${rowNumber + 2};`;
                      const colStyle = `grid-column:${colNumber + 2};`;
                      if (modelIndex === otherIndex) {
                        return `
                          <div
                            class="cell diag"
                            style="${rowStyle}${colStyle}"
                          >
                            <span class="diag-dot"></span>
                          </div>
                        `;
                      }
                      const pValue = pairwiseData.matrix.p_value[modelIndex]?.[otherIndex];
                      const direction = pairDirection(pairwiseData, modelIndex, otherIndex);
                      const diff = pairwiseData.matrix.score_diff[modelIndex]?.[otherIndex];
                      const style = cellColor(direction, pValue, state.alpha);
                      const diffLabel = fmtScoreDiff(diff, pairwiseData.meta, 1);
                      const signalLevel = cellSignalLevel(pValue, state.alpha);
                      const cellStyle = [
                        rowStyle,
                        colStyle,
                        `background:${style.bg};`,
                        `color:${style.fg};`,
                        `border-color:${style.border ?? "var(--c-rule)"};`,
                      ].join("");
                      const cellContent = `
                        <span class="cell-diff">${diffLabel}</span>
                        ${renderCellSignal(signalLevel)}
                      `;
                      return `
                        <div
                          class="cell"
                          style="${cellStyle}"
                          data-row="${modelIndex}"
                          data-col="${otherIndex}"
                        >
                          <span class="cell-inner">
                            ${cellContent}
                          </span>
                        </div>
                      `;
                    }).join("")}
                    <div class="summary-cell" style="grid-row:${rowNumber + 2};">
                      <span class="sum-w">${summary.wins}</span>
                      <span class="sum-sep">/</span>
                      <span class="sum-l">${summary.losses}</span>
                      <span class="sum-sep">/</span>
                      <span class="sum-n">${summary.ties}</span>
                    </div>
                  `;
                }).join("")}
              </div>
            </div>
          </div>
        `;
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

      function scopeCoverageSummary(groupData) {
        const resultsData = groupData?.results_table;
        if (!resultsData || !Array.isArray(resultsData.models)) return null;
        const scopedColumns = scopedTaskColumns(resultsData);
        const scoredRows = resultsData.models.filter((model) =>
          scopedColumns.some((column) => isNumber(model.task_scores?.[column.id]))
        ).length;
        return {
          groupModelCount: resultsData.models.length,
          scopeTaskCount: scopedColumns.length,
          scoredRows,
        };
      }

      function renderDiagnosticCounts(counts) {
        if (!Array.isArray(counts) || counts.length === 0) return "";
        return `
          <div class="diag-count-grid">
            ${counts.map((count) => `
              <div class="diag-count">
                <div class="diag-count-value">${esc(count.value)}</div>
                <div class="diag-count-label">${esc(count.label)}</div>
              </div>
            `).join("")}
          </div>
        `;
      }

      function renderDiagnosticBulletSection(title, items) {
        if (!Array.isArray(items) || items.length === 0) return "";
        return `
          <section class="diag-section">
            <div class="diag-section-title">${esc(title)}</div>
            <ul class="diag-bullets">
              ${items.map((item) => `<li>${esc(item)}</li>`).join("")}
            </ul>
          </section>
        `;
      }

      function renderDiagnosticRunSection(title, items) {
        if (!Array.isArray(items) || items.length === 0) return "";
        return `
          <section class="diag-section">
            <div class="diag-section-title">${esc(title)}</div>
            <div class="diag-list">
              ${items.map((item) => {
                const metaParts = [];
                if (item.timestamp_label) metaParts.push(item.timestamp_label);
                if (item.reason_summary) metaParts.push(item.reason_summary);
                if (item.experiment_id) metaParts.push(item.experiment_id);
                return `
                  <div class="diag-list-item">
                    <div class="diag-list-title">${esc(item.label || item.model_name || "unnamed run")}</div>
                    ${metaParts.length
                      ? `<div class="diag-list-meta">${esc(metaParts.join(" · "))}</div>`
                      : ""}
                  </div>
                `;
              }).join("")}
            </div>
          </section>
        `;
      }

      function renderDiagnosticInstanceCounts(title, items) {
        if (!Array.isArray(items) || items.length === 0) return "";
        return `
          <section class="diag-section">
            <div class="diag-section-title">${esc(title)}</div>
            <div class="diag-list diag-list-compact">
              ${items.map((item) => `
                <div class="diag-list-item diag-list-item-split">
                  <div class="diag-list-title">${esc(item.label || "")}</div>
                  <div class="diag-list-meta">${esc(item.instance_count)}</div>
                </div>
              `).join("")}
            </div>
          </section>
        `;
      }

      function renderPairwiseError(errorDetails, errorMessage, groupData) {
        const details = errorDetails || {};
        const coverage = scopeCoverageSummary(groupData);
        const scopeOption = selectedScopeOption();
        const metricOptions = Array.isArray(scopeOption?.metric_options)
          ? scopeOption.metric_options
          : [];
        const summary = details.summary || errorMessage || "paired test unavailable";
        const notes = Array.isArray(details.notes) ? details.notes.slice() : [];
        if (
          metricOptions.length > 0 &&
          (details.code === "missing_primary_metric" ||
            details.code === "insufficient_extractable_instance_scores")
        ) {
          notes.unshift(
            "This scope has stored metric values. Use the metric control above to choose a metric and retry the paired test."
          );
        }
        if (coverage) {
          notes.unshift(
            `The results tab is broader: it currently shows ${coverage.groupModelCount} latest group-level model row(s), with ${coverage.scoredRows} carrying at least one score in this scope. The paired test is stricter and only uses runs that match the scope, survive dedupe / coverage filtering, expose pairwise-eligible instance scores, and share common instances.`
          );
        }
        return `
          <div class="matrix-wrap">
            <div class="diag-card">
              <div class="diag-kicker">paired test unavailable</div>
              <div class="diag-title">${esc(summary)}</div>
              ${details.scope_label
                ? `<div class="diag-scope">${esc(details.scope_label)}</div>`
                : ""}
              ${renderDiagnosticCounts(details.counts)}
              ${renderDiagnosticBulletSection("why this can happen", notes)}
              ${details.filter_summary
                ? `
                  <section class="diag-section">
                    <div class="diag-section-title">active filters</div>
                    <div class="diag-code">${esc(details.filter_summary)}</div>
                  </section>
                `
                : ""}
              ${renderDiagnosticRunSection("runs that matched this scope", details.matched_runs)}
              ${renderDiagnosticRunSection("models still in the paired test pipeline", details.compared_models)}
              ${renderDiagnosticRunSection("older duplicate runs ignored", details.dropped_duplicate_runs)}
              ${renderDiagnosticRunSection(
                "dropped by the full-coverage requirement",
                details.dropped_partial_coverage_models
              )}
              ${renderDiagnosticRunSection("models with pairwise-eligible instance scores", details.scored_models)}
              ${renderDiagnosticRunSection("models missing pairwise-eligible instance scores", details.unscored_models)}
              ${renderDiagnosticBulletSection("unsupported task metrics", details.unsupported_task_metrics)}
              ${renderDiagnosticInstanceCounts("per-model instance counts", details.per_model_instance_counts)}
              ${renderDiagnosticBulletSection("what to do next", details.suggestions)}
            </div>
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
            <span
              class="swatch"
              style="background:${config.bg};color:${config.fg};"
            >
              ${config.mark}
            </span>
            <span>${esc(label)}</span>
          </span>
        `;
      }

      function sortPill(kind, label) {
        return `
          <button
            class="${state.matrixSort === kind ? "pill on" : "pill"}"
            data-action="matrix-sort"
            data-kind="${kind}"
          >
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

      function renderPairwiseExportMenu() {
        return `
          <div class="legend-group">
            <details class="tt-dd">
              <summary class="tt-icon-btn">
                ${downloadSvg()} export
              </summary>
              <div class="tt-menu">
                <div class="tt-menu-head"><span>data</span></div>
                <div class="tt-menu-body">
                  <button type="button" class="tt-menu-action" data-action="export-pairwise-csv">
                    <span class="tt-menu-name">paired comparisons</span>
                    <span class="tt-menu-n">csv</span>
                  </button>
                  <button type="button" class="tt-menu-action" data-action="export-pairwise-json">
                    <span class="tt-menu-name">paired test summary</span>
                    <span class="tt-menu-n">json</span>
                  </button>
                  <button
                    type="button"
                    class="tt-menu-action"
                    data-action="export-pairwise-instance-results"
                  >
                    <span class="tt-menu-name">instance results</span>
                    <span class="tt-menu-n">csv</span>
                  </button>
                  <button
                    type="button"
                    class="tt-menu-action"
                    data-action="export-pairwise-stored-files"
                  >
                    <span class="tt-menu-name">stored files</span>
                    <span class="tt-menu-n">csv</span>
                  </button>
                  <button type="button" class="tt-menu-action" data-action="export-pairwise-all">
                    <span class="tt-menu-name">all viewer data</span>
                    <span class="tt-menu-n">json</span>
                  </button>
                </div>
              </div>
            </details>
          </div>
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

      function renderApp() {
        const groupData = pageData.group_data;
        const pairwiseData = pageData.pairwise_data;
        root.innerHTML = state.view === "matrix"
          ? renderMatrix(pairwiseData, pageData.pairwise_error, pageData.pairwise_error_details)
          : renderResults(groupData);
        syncChrome();
        bindResultsScrollbars();
        bindColumnsMenu();
        queueResultsScrollSync();
        window.setTimeout(queueResultsScrollSync, 0);
        window.setTimeout(queueResultsScrollSync, 80);
      }

      function syncChrome() {
        const viewTable = document.getElementById("view-table");
        const viewMatrix = document.getElementById("view-matrix");
        if (viewTable) viewTable.classList.toggle("active", state.view === "table");
        if (viewMatrix) viewMatrix.classList.toggle("active", state.view === "matrix");
        syncModelFilterControls();
        const regexCount = document.getElementById("regex-count");
        const regexWrap = document.getElementById("regex-wrap");
        const compiled = compileRegex();
        if (regexWrap) regexWrap.classList.toggle("err", compiled.error);
        if (!regexCount) return;
        if (state.view === "matrix") {
          const filtered = filteredPairwiseModelIndices(pageData.pairwise_data);
          const total = pageData.pairwise_data ? pageData.pairwise_data.models.length : 0;
          regexCount.textContent = compiled.error
            ? "invalid"
            : `${filtered.indices.length}/${total}`;
        } else {
          const resultsData = pageData.group_data?.results_table || { models: [] };
          const filtered = filteredResultsModelIndices(resultsData);
          regexCount.textContent = compiled.error
            ? "invalid"
            : filtered.indices.length + "/" + resultsData.models.length;
        }
      }

      function syncModelFilterControls() {
        trimExcludedModels();
        const models = allFilterModels();
        const total = models.length;
        const visible = models.filter((model) => !isExcludedModel(model)).length;
        const summary = document.getElementById("model-filter-summary");
        const count = document.getElementById("model-filter-count");
        const reset = document.getElementById("model-filter-reset");
        document.querySelectorAll('input[data-action="toggle-model-checkbox"]').forEach((input) => {
          input.checked = !state.excludedModels.has(String(input.dataset.modelKey || ""));
        });
        if (summary) {
          if (total === 0) summary.textContent = "no models";
          else if (visible === total) summary.textContent = "all models";
          else if (visible === 0) summary.textContent = "all excluded";
          else summary.textContent = `${total - visible} excluded`;
        }
        if (count) {
          count.textContent = total === 0
            ? "0"
            : visible === total
              ? String(total)
              : `${visible}/${total}`;
        }
        if (reset) {
          reset.hidden = total === 0 || visible === total;
        }
      }

      function showTooltip(row, col, event) {
        const pairwiseData = pageData.pairwise_data;
        if (!pairwiseData) return;
        const tooltip = document.getElementById("pairwise-tooltip");
        const left = pairwiseData.models[row];
        const right = pairwiseData.models[col];
        const diff = pairwiseData.matrix.score_diff[row]?.[col];
        const pValue = pairwiseData.matrix.p_value[row]?.[col];
        const probability = pairwiseData.matrix.probability[row]?.[col];
        const wins = pairwiseData.matrix.wins[row]?.[col] ?? "-";
        const losses = pairwiseData.matrix.losses[row]?.[col] ?? "-";
        const ties = pairwiseData.matrix.ties[row]?.[col] ?? "-";
        const winRate = pairwiseData.matrix.win_rate[row]?.[col];
        const se = pairwiseData.matrix.se[row]?.[col];
        const direction = pairDirection(pairwiseData, row, col);
        const deltaClass = direction > 0 ? "pos" : direction < 0 ? "neg" : "";
        const pClass = isNumber(pValue) && pValue <= state.alpha ? "sig" : "ns";
        const taskCount = pairwiseData.meta.task_count ?? 0;
        const taskLabel = taskCount === 1 ? "task" : "tasks";
        const scopeLabel = pairwiseData.meta.scope_kind === "suite" && taskCount > 0
          ? `${pairwiseData.meta.scope_label} (${taskCount} ${taskLabel})`
          : pairwiseData.meta.scope_label;
        tooltip.innerHTML = `
          <div class="tt-head">
            <div class="tt-title">paired test</div>
            <div class="tt-sub">
              ${esc(scopeLabel)} · N=${pairwiseData.meta.shared_n}
            </div>
          </div>
          <div class="tt-pair">
            <div class="tt-row">
              <span class="tt-dot a"></span>
              <span class="tt-name">${esc(left.display_label)}</span>
              <span class="tt-acc">${fmtScoreValue(displayScore(left), pairwiseData.meta, 2)}</span>
            </div>
            <div class="tt-row">
              <span class="tt-dot b"></span>
              <span class="tt-name">${esc(right.display_label)}</span>
              <span class="tt-acc">${fmtScoreValue(displayScore(right), pairwiseData.meta, 2)}</span>
            </div>
          </div>
          <div class="tt-stats">
            <div class="tt-stat">
              <span class="k">Δ (row − col)</span>
              <span class="v ${deltaClass}">${fmtDelta(diff, pairwiseData.meta, 2)}</span>
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
          if (nodeRow === row && nodeCol === col) node.classList.add("hover");
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

      function slugify(value, fallback = "export") {
        const slug = String(value ?? "")
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-+|-+$/g, "");
        return slug || fallback;
      }

      function csvEscape(value) {
        const text = String(value ?? "");
        return /[,"\\n]/.test(text)
          ? '"' + text.replace(/"/g, '""') + '"'
          : text;
      }

      function downloadText(filename, contents, type) {
        const blob = new Blob([contents], { type });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      }

      function currentGroupName() {
        return pageData.group_data?.summary?.group_name || "results";
      }

      function pairwiseExportBase(pairwiseData) {
        const groupKey = slugify(currentGroupName(), "results");
        const scopeKey = slugify(
          pairwiseData?.meta?.storage_key ||
          pairwiseData?.meta?.scope_label ||
          pageData.selected_scope_key ||
          "pairwise",
          "pairwise"
        );
        return groupKey === scopeKey ? groupKey : `${groupKey}-${scopeKey}`;
      }

      function modelExportRef(model) {
        return String((model?.model_hash || "") + "|" + (model?.timestamp || ""));
      }

      function pairwiseExportModels(pairwiseData) {
        if (!pairwiseData) return [];
        const order = currentPairwiseExportOrder(pairwiseData);
        return order.map((modelIndex) => pairwiseData.models[modelIndex]).filter(Boolean);
      }

      function pairwiseViewerExportUrl(kind, format = "csv") {
        const pairwiseData = pageData.pairwise_data;
        const url = new URL("/export", window.location.origin);
        url.searchParams.set("kind", kind);
        url.searchParams.set("format", format);
        const groupName = currentGroupName();
        if (groupName) url.searchParams.set("group", groupName);
        if (pageData.selected_scope_key) url.searchParams.set("scope", pageData.selected_scope_key);
        if (pageData.selected_metric) url.searchParams.set("metric", pageData.selected_metric);
        if (pageData.selected_run_mode) url.searchParams.set("runs", pageData.selected_run_mode);
        pairwiseExportModels(pairwiseData).forEach((model) => {
          url.searchParams.append("model_ref", modelExportRef(model));
        });
        return url;
      }

      function triggerDownload(url) {
        const link = document.createElement("a");
        link.href = url.toString();
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }

      async function fetchViewerExportJson(kind) {
        const response = await fetch(pairwiseViewerExportUrl(kind, "json"));
        if (!response.ok) {
          const message = (await response.text()).trim();
          throw new Error(message || "The viewer export failed.");
        }
        return response.json();
      }

      function currentPairwiseExportOrder(pairwiseData) {
        if (!pairwiseData) return [];
        const filtered = filteredPairwiseModelIndices(pairwiseData).indices;
        return matrixOrder(pairwiseData, filtered);
      }

      function buildPairwiseExportRows(
        pairwiseData,
        order = currentPairwiseExportOrder(pairwiseData)
      ) {
        if (!pairwiseData) return [];
        const rows = [];
        for (let rowNumber = 0; rowNumber < order.length; rowNumber += 1) {
          const rowIndex = order[rowNumber];
          const rowModel = pairwiseData.models[rowIndex];
          for (let colNumber = rowNumber + 1; colNumber < order.length; colNumber += 1) {
            const colIndex = order[colNumber];
            const colModel = pairwiseData.models[colIndex];
            const pValue = pairwiseData.matrix.p_value[rowIndex]?.[colIndex];
            const scoreDiff = pairwiseData.matrix.score_diff[rowIndex]?.[colIndex];
            rows.push({
              row_display_rank: rowNumber + 1,
              column_display_rank: colNumber + 1,
              row_model_index: rowIndex,
              column_model_index: colIndex,
              row_model_label: rowModel?.display_label || "",
              column_model_label: colModel?.display_label || "",
              row_win_count: pairwiseData.matrix.wins[rowIndex]?.[colIndex] ?? null,
              column_win_count: pairwiseData.matrix.wins[colIndex]?.[rowIndex] ?? null,
              tie_count: pairwiseData.matrix.ties[rowIndex]?.[colIndex] ?? null,
              contested_instance_count: pairwiseData.matrix.contested[rowIndex]?.[colIndex] ?? null,
              row_win_rate: pairwiseData.matrix.win_rate[rowIndex]?.[colIndex] ?? null,
              column_win_rate: pairwiseData.matrix.win_rate[colIndex]?.[rowIndex] ?? null,
              win_rate_standard_error: pairwiseData.matrix.se[rowIndex]?.[colIndex] ?? null,
              probability_row_beats_column:
                pairwiseData.matrix.probability[rowIndex]?.[colIndex] ?? null,
              p_value: pValue ?? null,
              score_difference_row_minus_column: scoreDiff ?? null,
              score_difference_display_format:
                pairwiseData.meta.score_display_format || "percentage",
              score_difference_unit: pairwiseData.meta.score_unit ?? null,
              significant_at_alpha: isNumber(pValue) ? pValue <= state.alpha : null,
              significance_alpha: state.alpha,
            });
          }
        }
        return rows;
      }

      function subsetSquareMatrix(matrix, order) {
        return order.map((rowIndex) =>
          order.map((colIndex) => matrix?.[rowIndex]?.[colIndex] ?? null)
        );
      }

      function buildPairwiseExportPayload(pairwiseData) {
        const order = currentPairwiseExportOrder(pairwiseData);
        const anchorVisible = order.includes(state.anchorIndex);
        const taskScoreColumns = Array.isArray(pairwiseData.task_columns)
          ? pairwiseData.task_columns.map((column) => ({
              task_name: column.full_label || column.id,
              task_label: column.label,
              score_display_format: column.score_display_format || "percentage",
              score_unit: column.score_unit ?? null,
              score_higher_is_better:
                column.higher_is_better === undefined ? null : column.higher_is_better,
            }))
          : [];
        const taskScoreSummary = Array.isArray(pairwiseData.task_stats)
          ? pairwiseData.task_stats.map((taskStat) => ({
              task_name: taskStat.full_label || taskStat.id,
              task_label: taskStat.label,
              median_model_score: taskStat.median_score ?? null,
              score_spread: taskStat.spread ?? null,
              best_model_label: taskStat.best_model_label ?? null,
              best_model_score: taskStat.best_model_score ?? null,
              worst_model_label: taskStat.worst_model_label ?? null,
              worst_model_score: taskStat.worst_model_score ?? null,
            }))
          : [];
        return {
          metadata: {
            page_title: pairwiseData.meta.title || null,
            group_name: currentGroupName(),
            scope_name: pairwiseData.meta.scope_label || null,
            scope_kind: pairwiseData.meta.scope_kind || null,
            selected_scope_key: pageData.selected_scope_key || null,
            metric_name: pairwiseData.meta.metric || null,
            selected_metric: pageData.selected_metric || null,
            run_selection: pageData.selected_run_mode || "latest",
            shared_instance_count: pairwiseData.meta.shared_n ?? null,
            task_count: pairwiseData.meta.task_count ?? null,
            compared_model_count: pairwiseData.meta.model_count ?? null,
            exported_model_count: order.length,
            tie_margin: pairwiseData.meta.margin ?? null,
            current_alpha: state.alpha,
            paired_test_sort: state.matrixSort,
            anchor_model_index: anchorVisible ? state.anchorIndex : null,
            anchor_model_label: anchorVisible
              ? pairwiseData.models[state.anchorIndex]?.display_label || null
              : null,
            mde80_at_default_alpha: pairwiseData.meta.mde80 ?? null,
            mde80_by_alpha: pairwiseData.meta.mde80_by_alpha || {},
            score_display_format: pairwiseData.meta.score_display_format || "percentage",
            score_unit: pairwiseData.meta.score_unit ?? null,
            score_higher_is_better:
              pairwiseData.meta.higher_is_better === undefined
                ? null
                : pairwiseData.meta.higher_is_better,
            score_scale_comparable:
              pairwiseData.meta.score_scale_comparable === undefined
                ? null
                : pairwiseData.meta.score_scale_comparable,
            matched_experiment_count: pairwiseData.meta.matched_experiments ?? null,
            dropped_experiment_count: pairwiseData.meta.dropped_experiments ?? null,
            has_costs:
              pairwiseData.meta.has_costs === undefined ? null : pairwiseData.meta.has_costs,
            storage_key: pairwiseData.meta.storage_key || null,
          },
          filters: {
            model_name_regex: state.regex,
            excluded_model_filter_keys: Array.from(state.excludedModels),
          },
          task_score_columns: taskScoreColumns,
          task_score_summary: taskScoreSummary,
          models: order.map((modelIndex, position) => {
            const model = pairwiseData.models[modelIndex] || {};
            return {
              display_rank: position + 1,
              model_index: model.index ?? modelIndex,
              display_label: model.display_label || "",
              model_name: model.model_name || null,
              model_hash: model.model_hash || null,
              model_hash_short: model.model_hash_short || null,
              timestamp: model.timestamp || null,
              shared_instance_mean_score: model.shared_score ?? null,
              mean_task_score: model.avg_task_score ?? null,
              bt_elo: model.strength ?? null,
              mean_pairwise_win_rate: model.avg_win_rate ?? null,
              significant_pairwise_net_wins: model.dominance ?? null,
              best_scoring_task_label: model.best_task_label ?? null,
              best_scoring_task_score: model.best_task_score ?? null,
              worst_scoring_task_label: model.worst_task_label ?? null,
              worst_scoring_task_score: model.worst_task_score ?? null,
              total_cost: model.cost ?? null,
              task_scores_by_task_name: model.task_scores || {},
            };
          }),
          pairwise_comparisons: buildPairwiseExportRows(pairwiseData, order),
          pairwise_matrices: {
            display_order_model_indices: order,
            display_order_model_labels: order.map(
              (modelIndex) => pairwiseData.models[modelIndex]?.display_label || ""
            ),
            row_win_count: subsetSquareMatrix(pairwiseData.matrix.wins, order),
            row_loss_count: subsetSquareMatrix(pairwiseData.matrix.losses, order),
            tie_count: subsetSquareMatrix(pairwiseData.matrix.ties, order),
            contested_instance_count: subsetSquareMatrix(pairwiseData.matrix.contested, order),
            row_win_rate: subsetSquareMatrix(pairwiseData.matrix.win_rate, order),
            win_rate_standard_error: subsetSquareMatrix(pairwiseData.matrix.se, order),
            probability_row_beats_column: subsetSquareMatrix(pairwiseData.matrix.probability, order),
            p_value: subsetSquareMatrix(pairwiseData.matrix.p_value, order),
            score_difference_row_minus_column: subsetSquareMatrix(
              pairwiseData.matrix.score_diff,
              order
            ),
          },
        };
      }

      function exportCsv() {
        const resultsData = pageData.group_data?.results_table;
        if (!resultsData) return;
        const filtered = filteredResultsModelIndices(resultsData).indices;
        const columns = visibleTaskColumns(resultsData);
        const showAverage = showAverageColumn(columns);
        const avgMeta = aggregateColumnMeta(columns);
        const sortState = resolvedTableSort(columns, showAverage);
        const rows = sortedTableRows(resultsData, filtered, columns, sortState);
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
          row.map((value) => csvEscape(value)).join(",")
        );
        downloadText(
          slugify(currentGroupName(), "results") + ".csv",
          lines.join("\\n") + "\\n",
          "text/csv;charset=utf-8"
        );
      }

      function exportPairwiseCsv() {
        const pairwiseData = pageData.pairwise_data;
        if (!pairwiseData) return;
        const rows = buildPairwiseExportRows(pairwiseData);
        const header = [
          "row_display_rank",
          "column_display_rank",
          "row_model_index",
          "column_model_index",
          "row_model_label",
          "column_model_label",
          "row_win_count",
          "column_win_count",
          "tie_count",
          "contested_instance_count",
          "row_win_rate",
          "column_win_rate",
          "win_rate_standard_error",
          "probability_row_beats_column",
          "p_value",
          "score_difference_row_minus_column",
          "score_difference_display_format",
          "score_difference_unit",
          "significant_at_alpha",
          "significance_alpha",
        ];
        const lines = [header, ...rows.map((row) => [
          row.row_display_rank,
          row.column_display_rank,
          row.row_model_index,
          row.column_model_index,
          row.row_model_label,
          row.column_model_label,
          row.row_win_count,
          row.column_win_count,
          row.tie_count,
          row.contested_instance_count,
          row.row_win_rate,
          row.column_win_rate,
          row.win_rate_standard_error,
          row.probability_row_beats_column,
          row.p_value,
          row.score_difference_row_minus_column,
          row.score_difference_display_format,
          row.score_difference_unit,
          row.significant_at_alpha,
          row.significance_alpha,
        ])].map((row) => row.map((value) => csvEscape(value)).join(","));
        downloadText(
          pairwiseExportBase(pairwiseData) + ".csv",
          lines.join("\\n") + "\\n",
          "text/csv;charset=utf-8"
        );
      }

      function exportPairwiseJson() {
        const pairwiseData = pageData.pairwise_data;
        if (!pairwiseData) return;
        const payload = buildPairwiseExportPayload(pairwiseData);
        downloadText(
          pairwiseExportBase(pairwiseData) + ".json",
          JSON.stringify(payload, null, 2) + "\\n",
          "application/json;charset=utf-8"
        );
      }

      function exportPairwiseInstanceResults() {
        triggerDownload(pairwiseViewerExportUrl("instance-results", "csv"));
      }

      function exportPairwiseStoredFiles() {
        triggerDownload(pairwiseViewerExportUrl("stored-files", "csv"));
      }

      async function exportPairwiseAll() {
        const pairwiseData = pageData.pairwise_data;
        if (!pairwiseData) return;
        try {
          const [instanceResults, storedFiles] = await Promise.all([
            fetchViewerExportJson("instance-results"),
            fetchViewerExportJson("stored-files"),
          ]);
          const payload = {
            paired_test: buildPairwiseExportPayload(pairwiseData),
            instance_results: instanceResults,
            stored_files: storedFiles,
          };
          downloadText(
            pairwiseExportBase(pairwiseData) + "-all.json",
            JSON.stringify(payload, null, 2) + "\\n",
            "application/json;charset=utf-8"
          );
        } catch (error) {
          window.alert(error instanceof Error ? error.message : "The viewer export failed.");
        }
      }

      scopeForm?.addEventListener("submit", (event) => {
        event.preventDefault();
        submitScopeForm();
      });

      scopeForm?.querySelectorAll("[data-search-select]").forEach((control) => {
        const details = control.querySelector(".search-select-dd");
        const filterInput = control.querySelector('[data-role="search-select-filter"]');
        const summary = details?.querySelector("summary");
        resetSearchSelect(control);
        syncSearchSelectMenuWidth(control);

        details?.addEventListener("toggle", () => {
          if (details.open) {
            closeSearchSelects(details);
            resetSearchSelect(control, true);
          } else {
            resetSearchSelect(control);
          }
        });

        filterInput?.addEventListener("input", (event) => {
          filterSearchSelect(control, event.target.value);
        });

        filterInput?.addEventListener("keydown", (event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            if (details) details.open = false;
            summary?.focus();
            return;
          }
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            moveActiveSearchOption(control, event.key === "ArrowDown" ? 1 : -1);
            return;
          }
          if (event.key === "Enter") {
            const option =
              activeSearchOption(control) ||
              (orderedVisibleSearchOptions(control).length === 1
                ? orderedVisibleSearchOptions(control)[0]
                : bestVisibleSearchOption(control, filterInput?.value));
            if (!option) return;
            event.preventDefault();
            option.click();
          }
        });

        control.querySelectorAll('[data-role="search-select-option"]').forEach((option) => {
          option.addEventListener("mouseenter", () => {
            if (!option.hidden) setActiveSearchOption(control, option);
          });
          option.addEventListener("focus", () => {
            if (!option.hidden) setActiveSearchOption(control, option);
          });
        });
      });

      scopeForm?.addEventListener("click", (event) => {
        const target = event.target.closest('[data-action="select-search-option"]');
        if (!target || !scopeForm.contains(target)) return;
        event.preventDefault();
        const control = target.closest("[data-search-select]");
        const details = control?.querySelector(".search-select-dd");
        const hiddenInput = control?.querySelector('input[type="hidden"][name]');
        if (!control || !hiddenInput) return;
        const nextValue = String(target.dataset.value || "");
        const previousValue = hiddenInput.value;
        hiddenInput.value = nextValue;
        setSearchSelectSummary(control, target);
        if (details) details.open = false;
        if (nextValue === previousValue) return;
        if (hiddenInput.name === "group" || hiddenInput.name === "scope") {
          const metricSelect = scopeForm.querySelector("#metric-select");
          if (metricSelect) metricSelect.value = "";
        }
        submitScopeForm();
      });

      document.addEventListener("click", (event) => {
        document.querySelectorAll(".search-select-dd[open]").forEach((details) => {
          if (!details.contains(event.target)) details.open = false;
        });
        const colsMenu = root.querySelector(".tt-cols-menu");
        if (colsMenu?.open && !colsMenu.contains(event.target)) {
          colsMenu.open = false;
          state.columnsMenuOpen = false;
        }
        if (modelFilterDetails?.open && !modelFilterDetails.contains(event.target)) {
          modelFilterDetails.open = false;
        }
      });

      document.addEventListener("pointermove", (event) => {
        if (!activeTableScrollDrag) return;
        const { region, maxOffset, maxScroll, startX, startScrollLeft } = activeTableScrollDrag;
        if (maxOffset <= 0 || maxScroll <= 0) return;
        const deltaX = event.clientX - startX;
        region.scrollLeft = clamp(
          startScrollLeft + (deltaX / maxOffset) * maxScroll,
          0,
          maxScroll
        );
      });

      function endTableScrollDrag() {
        if (!activeTableScrollDrag) return;
        activeTableScrollDrag = null;
        document.body.classList.remove("is-dragging-table-xbar");
      }

      document.addEventListener("pointerup", endTableScrollDrag);
      document.addEventListener("pointercancel", endTableScrollDrag);
      window.addEventListener("resize", queueResultsScrollSync);
      window.addEventListener("resize", queueSearchSelectMenuWidthSync);
      document.fonts?.ready?.then(() => {
        queueResultsScrollSync();
        queueSearchSelectMenuWidthSync();
      });

      document.getElementById("view-table")?.addEventListener("click", () => {
        state.view = "table";
        persistState();
        renderApp();
      });

      document.getElementById("view-matrix")?.addEventListener("click", () => {
        state.view = "matrix";
        persistState();
        renderApp();
      });

      document.getElementById("regex-filter")?.addEventListener("input", (event) => {
        state.regex = event.target.value;
        persistState();
        renderApp();
      });

      document.getElementById("metric-select")?.addEventListener("change", () => {
        submitScopeForm();
      });

      document.getElementById("run-mode-select")?.addEventListener("change", () => {
        submitScopeForm();
      });

      document.querySelectorAll('input[data-action="toggle-model-checkbox"]').forEach((input) => {
        input.addEventListener("change", (event) => {
          const target = event.target;
          const key = String(target.dataset.modelKey || "");
          if (!key) return;
          if (target.checked) state.excludedModels.delete(key);
          else state.excludedModels.add(key);
          persistState();
          renderApp();
        });
      });

      document.getElementById("model-filter-reset")?.addEventListener("click", () => {
        state.excludedModels.clear();
        persistState();
        renderApp();
      });

      root.addEventListener("click", (event) => {
        const target = event.target.closest("[data-action]");
        if (!target) return;
        const action = target.dataset.action;
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
        if (action === "exclude-model") {
          const key = String(target.dataset.modelKey || "");
          if (!key) return;
          state.excludedModels.add(key);
          persistState();
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
        if (action === "solo-col") {
          const id = target.dataset.id;
          const resultsData = pageData.group_data?.results_table;
          if (!id || !resultsData) return;
          state.columnsMenuOpen = true;
          state.hiddenCols = new Set(
            scopedTaskColumns(resultsData)
              .map((column) => column.id)
              .filter((columnId) => columnId !== id)
          );
          renderApp();
          return;
        }
        if (action === "reset-cols") {
          const resultsData = pageData.group_data?.results_table;
          if (!resultsData) return;
          state.columnsMenuOpen = true;
          scopedTaskColumns(resultsData).forEach((column) => {
            state.hiddenCols.delete(column.id);
          });
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
              const resultsTable = pageData.group_data?.results_table;
              const visibleColumns = resultsTable ? visibleTaskColumns(resultsTable) : [];
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
          return;
        }
        if (action === "export-pairwise-csv") {
          target.closest("details")?.removeAttribute("open");
          exportPairwiseCsv();
          return;
        }
        if (action === "export-pairwise-json") {
          target.closest("details")?.removeAttribute("open");
          exportPairwiseJson();
          return;
        }
        if (action === "export-pairwise-instance-results") {
          target.closest("details")?.removeAttribute("open");
          exportPairwiseInstanceResults();
          return;
        }
        if (action === "export-pairwise-stored-files") {
          target.closest("details")?.removeAttribute("open");
          exportPairwiseStoredFiles();
          return;
        }
        if (action === "export-pairwise-all") {
          target.closest("details")?.removeAttribute("open");
          void exportPairwiseAll();
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
        if (action === "toggle-col-checkbox") {
          const id = target.dataset.id;
          state.columnsMenuOpen = true;
          if (target.checked) state.hiddenCols.delete(id);
          else state.hiddenCols.add(id);
          renderApp();
        }
      });

      root.addEventListener("mouseover", (event) => {
        const cell = event.target.closest(".cell[data-row][data-col]");
        if (!cell || !root.contains(cell)) return;
        showTooltip(parseInt(cell.dataset.row, 10), parseInt(cell.dataset.col, 10), event);
      });

      root.addEventListener("mousemove", (event) => {
        if (hoverPair !== null) positionTooltip(event);
      });

      root.addEventListener("mouseout", (event) => {
        const cell = event.target.closest(".cell[data-row][data-col]");
        if (!cell) return;
        const related = event.relatedTarget;
        if (related && cell.contains(related)) return;
        hideTooltip();
      });

      trimExcludedModels();
      queueSearchSelectMenuWidthSync();
      renderApp();
    })();
