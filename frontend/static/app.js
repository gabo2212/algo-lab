const ICONS = {
  check: `<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M7.7 13.3 4.4 10l-1.4 1.4 4.7 4.7 10-10-1.4-1.4z"/></svg>`,
};

const IDENTITY = {
  name: "King BBC",
  pfp: "/static/king-bbc.webp",
};

function pfpAvatar(kind) {
  return `<div class="avatar ${kind}"><img src="${IDENTITY.pfp}" alt="${IDENTITY.name}" /></div>`;
}

const state = {
  algorithms: [],
  current: null,
  chats: {},
  details: {},
  busy: false,
};

const els = {
  channelList: document.getElementById("channel-list"),
  modelOptions: document.getElementById("model-options"),
  modelTrigger: document.getElementById("model-trigger"),
  modelMenu: document.getElementById("model-menu"),
  modelLabel: document.getElementById("model-label"),
  headerTopic: document.getElementById("header-topic"),
  messages: document.getElementById("messages"),
  featurePanel: document.getElementById("feature-panel"),
  composer: document.getElementById("composer"),
  sendBtn: document.getElementById("send-btn"),
  plusBtn: document.getElementById("plus-btn"),
  plusMenu: document.getElementById("plus-menu"),
  howToRead: document.getElementById("how-to-read"),
  metricsBox: document.getElementById("metrics-box"),
  apiDot: document.getElementById("api-dot"),
  sidebar: document.getElementById("channel-sidebar"),
  backdrop: document.getElementById("backdrop"),
  openSidebar: document.getElementById("open-sidebar"),
  closeSidebar: document.getElementById("close-sidebar"),
  uploadBtn: document.getElementById("upload-btn"),
  csvInput: document.getElementById("csv-input"),
  csvHint: document.getElementById("csv-hint"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function kindLabel(kind) {
  const map = {
    regression: "Régression",
    "classification-binaire": "Classification binaire",
    "classification-multiclasse": "Classification multiclasse",
    "classification-proximite": "Classification par proximité",
    "classification-ensemble": "Classification par ensemble",
  };
  return map[kind] || kind;
}

function nowStamp() {
  return new Intl.DateTimeFormat("fr-CA", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date());
}

function channelName(algo) {
  return algo.slug.replaceAll("-", "-");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function closeMenus() {
  els.modelMenu.hidden = true;
  els.modelTrigger.setAttribute("aria-expanded", "false");
  els.plusMenu.hidden = true;
  els.plusBtn.setAttribute("aria-expanded", "false");
}

function renderChannels() {
  els.channelList.innerHTML = "";
  state.algorithms.forEach((algo) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = `channel-item${state.current === algo.slug ? " active" : ""}`;
    button.innerHTML = `<span class="hash">#</span><span>${channelName(algo)}</span>`;
    button.addEventListener("click", () => selectAlgorithm(algo.slug));
    item.append(button);
    els.channelList.append(item);
  });
}

function renderDropdown() {
  els.modelOptions.innerHTML = "";
  state.algorithms.forEach((algo) => {
    const selected = state.current === algo.slug;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-option";
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", String(selected));
    button.innerHTML = `
      <span class="hash">#</span>
      <span>
        <strong>${algo.name}</strong>
        <span class="kind">${kindLabel(algo.kind)}</span>
      </span>
      <span class="check">${selected ? ICONS.check : ""}</span>
    `;
    button.addEventListener("click", () => {
      selectAlgorithm(algo.slug);
      closeMenus();
    });
    els.modelOptions.append(button);
  });
}

function hyperControls(algo) {
  const keys = Object.keys(algo.hyperparameters || {});
  if (!keys.length) return "";
  return keys
    .map((key) => {
      if (key === "kernel") {
        const value = algo.hyperparameters.kernel || "rbf";
        return `<div class="field"><label for="hp-${key}">Noyau</label>
          <select id="hp-${key}" name="${key}">
            <option value="rbf" ${value === "rbf" ? "selected" : ""}>rbf</option>
            <option value="linear" ${value === "linear" ? "selected" : ""}>linear</option>
          </select></div>`;
      }
      const value = algo.hyperparameters[key];
      return `<div class="field"><label for="hp-${key}">${key}</label>
        <input id="hp-${key}" name="${key}" type="number" value="${value}" /></div>`;
    })
    .join("");
}

function fieldNameSelector(name) {
  return `[name="${CSS.escape(name)}"]`;
}

function fieldId(name) {
  return `feat-${name.replace(/[^a-z0-9]+/gi, "-")}`;
}

function renderComposer(algo) {
  const sample = algo.sample_input || {};
  const featureFields = algo.feature_names
    .map((name) => {
      const value = sample[name] ?? "";
      const id = fieldId(name);
      return `<div class="field">
        <label for="${id}">${escapeHtml(name)}</label>
        <input id="${id}" name="${escapeHtml(name)}" type="number" step="any" value="${escapeHtml(value)}" required />
      </div>`;
    })
    .join("");
  els.featurePanel.innerHTML = featureFields + hyperControls(algo);
  if (els.csvHint) {
    const preview = algo.feature_names.slice(0, 4).join(", ");
    const extra = algo.feature_names.length > 4 ? "…" : "";
    els.csvHint.textContent = `CSV : ${preview}${extra}`;
  }
}

function fillFeatures(values) {
  Object.entries(values).forEach(([name, value]) => {
    const input = els.featurePanel.querySelector(fieldNameSelector(name));
    if (input) input.value = value;
  });
}

function collectFeatures(algo) {
  const features = {};
  algo.feature_names.forEach((name) => {
    const input = els.featurePanel.querySelector(fieldNameSelector(name));
    features[name] = Number(input.value);
  });
  return features;
}

function collectHyperparameters(algo) {
  const hyperparameters = {};
  Object.keys(algo.hyperparameters || {}).forEach((key) => {
    const input = els.featurePanel.querySelector(`[name="${key}"]`);
    if (!input) return;
    hyperparameters[key] = key === "kernel" ? input.value : Number(input.value);
  });
  return hyperparameters;
}

function formatMetrics(metrics) {
  if (!metrics) return "Aucune métrique.";
  const lines = [];
  if (metrics.mae != null) lines.push(`MAE  ${metrics.mae}`);
  if (metrics.r2 != null) lines.push(`R²   ${metrics.r2}`);
  if (metrics.accuracy != null) lines.push(`Exactitude  ${metrics.accuracy}`);
  if (metrics.top_features) {
    lines.push("Top caractéristiques:");
    metrics.top_features.forEach((row) => {
      lines.push(`  ${row.caracteristique}: ${Number(row.importance).toFixed(3)}`);
    });
  }
  if (metrics.test_example) {
    lines.push(`Exemple test: ${metrics.test_example.real_class} → ${metrics.test_example.predicted_class}`);
  }
  return lines.join("\n") || JSON.stringify(metrics, null, 2);
}

function predictionLines(prediction) {
  if (!prediction) return [];
  const rows = [];
  if (prediction.predicted_price != null) {
    rows.push(["Prix prédit", `${prediction.predicted_price} ${prediction.unit || ""}`]);
  }
  if (prediction.result) rows.push(["Résultat", prediction.result]);
  if (prediction.success_probability_percent != null) {
    rows.push(["Probabilité", `${prediction.success_probability_percent} %`]);
  }
  if (prediction.predicted_species) rows.push(["Espèce", prediction.predicted_species]);
  if (prediction.predicted_category) rows.push(["Catégorie", prediction.predicted_category]);
  if (prediction.predicted_label) rows.push(["Classe", prediction.predicted_label]);
  if (prediction.probabilities_percent) {
    Object.entries(prediction.probabilities_percent).forEach(([name, value]) => {
      rows.push([name, `${value} %`]);
    });
  }
  if (prediction.disclaimer) rows.push(["Note", prediction.disclaimer]);
  return rows;
}

function csvTable(rows) {
  if (!rows?.length) return "";
  const body = rows
    .map(
      (row) => `<tr>
        <td>${row.row}</td>
        <td>${escapeHtml(row.summary)}</td>
        <td>${row.actual == null ? "—" : escapeHtml(row.actual)}</td>
      </tr>`
    )
    .join("");
  return `<div class="result-table-wrap"><table class="result-table">
    <thead><tr><th>Ligne</th><th>Prédiction</th><th>Réel</th></tr></thead>
    <tbody>${body}</tbody>
  </table></div>`;
}

function renderMessages() {
  const chat = state.chats[state.current] || [];
  els.messages.innerHTML = chat
    .map((msg) => {
      if (msg.type === "typing") {
        return `<article class="message">
          ${pfpAvatar("tutor")}
          <div><div class="typing"><span></span><span></span><span></span></div></div>
        </article>`;
      }
      const isUser = msg.role === "user";
      const kv = (msg.fields || [])
        .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`)
        .join("");
      const actions = (msg.actions || [])
        .map(
          (action) =>
            `<button class="chip-btn ${action.style || ""}" type="button" data-action="${action.id}">${escapeHtml(action.label)}</button>`
        )
        .join("");
      return `<article class="message">
        ${pfpAvatar(isUser ? "user" : "tutor")}
        <div>
          <div class="msg-head">
            <strong>${IDENTITY.name} ${isUser ? "" : '<span class="bot-badge">BOT</span>'}</strong>
            <span class="timestamp">aujourd’hui à ${msg.time}</span>
          </div>
          ${msg.text ? `<p>${escapeHtml(msg.text)}</p>` : ""}
          ${
            msg.embed
              ? `<div class="embed ${msg.embedTone || ""}">
                  ${msg.embedTitle ? `<h3>${escapeHtml(msg.embedTitle)}</h3>` : ""}
                  ${msg.embedText ? `<p>${escapeHtml(msg.embedText)}</p>` : ""}
                  ${kv ? `<dl class="kv">${kv}</dl>` : ""}
                  ${msg.tableHtml || ""}
                  ${actions ? `<div class="action-row">${actions}</div>` : ""}
                </div>`
              : ""
          }
        </div>
      </article>`;
    })
    .join("");
  scrollChatToBottom();
}

function scrollChatToBottom() {
  const pane = els.messages;
  const pin = () => {
    pane.scrollTop = pane.scrollHeight;
    pane.lastElementChild?.scrollIntoView({ block: "end", inline: "nearest" });
  };
  pin();
  requestAnimationFrame(() => {
    pin();
    requestAnimationFrame(pin);
  });
}

function pushMessage(slug, message) {
  state.chats[slug] = state.chats[slug] || [];
  state.chats[slug].push({ time: nowStamp(), ...message });
  if (slug === state.current) renderMessages();
}

function setTyping(on) {
  const chat = state.chats[state.current] || [];
  state.chats[state.current] = chat.filter((msg) => msg.type !== "typing");
  if (on) state.chats[state.current].push({ type: "typing" });
  renderMessages();
}

function seedWelcome(algo, detail) {
  if (state.chats[algo.slug]?.length) return;
  pushMessage(algo.slug, {
    role: "bot",
    embed: true,
    embedTone: "",
    embedTitle: algo.name,
    embedText: `${algo.objective} ${algo.principle}`,
    fields: [
      ["Type", kindLabel(algo.kind)],
      ["Fichier", algo.file_name],
    ],
    actions: [
      { id: "example", label: "Lancer l’exemple" },
      { id: "exercise", label: "À vous de jouer", style: "secondary" },
      { id: "upload", label: "Importer un CSV", style: "secondary" },
    ],
  });
  if (detail?.metrics) {
    const metrics = detail.metrics;
    const fields = [];
    if (metrics.mae != null) fields.push(["MAE", metrics.mae]);
    if (metrics.r2 != null) fields.push(["R²", metrics.r2]);
    if (metrics.accuracy != null) fields.push(["Exactitude", metrics.accuracy]);
    pushMessage(algo.slug, {
      role: "bot",
      text: "Le modèle est déjà entraîné. Voici les métriques du jeu de test.",
      embed: true,
      embedTitle: "Évaluation",
      fields,
    });
  }
}

function updateSidebar(algo, detail) {
  els.howToRead.textContent = algo.how_to_read;
  els.metricsBox.textContent = formatMetrics(detail?.metrics);
}

async function selectAlgorithm(slug) {
  const algo = state.algorithms.find((item) => item.slug === slug);
  if (!algo) return;
  state.current = slug;
  els.modelLabel.textContent = algo.name;
  els.headerTopic.textContent = algo.objective;
  renderChannels();
  renderDropdown();
  renderComposer(algo);
  els.sidebar.classList.remove("open");
  els.backdrop.hidden = true;

  if (!state.details[slug]) {
    state.details[slug] = await api(`/algorithms/${slug}`);
  }
  const detail = state.details[slug];
  Object.assign(algo, detail);
  renderComposer(algo);
  seedWelcome(algo, detail);
  updateSidebar(algo, detail);
  renderMessages();
}

async function runPredict() {
  const algo = state.algorithms.find((item) => item.slug === state.current);
  if (!algo || state.busy) return;
  const features = collectFeatures(algo);
  const hyperparameters = collectHyperparameters(algo);
  pushMessage(algo.slug, {
    role: "user",
    text: "Nouvelle observation à prédire.",
    embed: true,
    fields: Object.entries(features).map(([k, v]) => [k, v]),
  });
  await requestPrediction(features, hyperparameters);
}

async function requestPrediction(features, hyperparameters) {
  const slug = state.current;
  state.busy = true;
  els.sendBtn.disabled = true;
  setTyping(true);
  try {
    const result = await api(`/algorithms/${slug}/predict`, {
      method: "POST",
      body: JSON.stringify({ features, hyperparameters }),
    });
    state.details[slug] = { ...(state.details[slug] || {}), ...result };
    setTyping(false);
    pushMessage(slug, {
      role: "bot",
      embed: true,
      embedTone: "success",
      embedTitle: "Prédiction",
      fields: predictionLines(result.prediction),
    });
    const algo = state.algorithms.find((item) => item.slug === slug);
    updateSidebar(algo, result);
  } catch (error) {
    setTyping(false);
    pushMessage(slug, {
      role: "bot",
      embed: true,
      embedTone: "error",
      embedTitle: "Impossible de prédire",
      embedText: error.message,
    });
  } finally {
    state.busy = false;
    els.sendBtn.disabled = false;
  }
}

async function runExercise() {
  const slug = state.current;
  const algo = state.algorithms.find((item) => item.slug === slug);
  if (!algo || state.busy) return;
  pushMessage(slug, {
    role: "user",
    text: `À vous de jouer : ${algo.exercise.prompt}`,
  });
  state.busy = true;
  els.sendBtn.disabled = true;
  setTyping(true);
  try {
    const result = await api(`/algorithms/${slug}/exercise`, { method: "POST" });
    setTyping(false);
    if (result.exercise?.input || algo.exercise.input) {
      fillFeatures(algo.exercise.input || {});
    }
    pushMessage(slug, {
      role: "bot",
      embed: true,
      embedTone: "success",
      embedTitle: "Exercice du devoir",
      embedText: result.prompt,
      fields: predictionLines(result.prediction),
    });
    if (result.train?.metrics) {
      updateSidebar(algo, { metrics: result.train.metrics });
    }
  } catch (error) {
    setTyping(false);
    pushMessage(slug, {
      role: "bot",
      embed: true,
      embedTone: "error",
      embedTitle: "Exercice impossible",
      embedText: error.message,
    });
  } finally {
    state.busy = false;
    els.sendBtn.disabled = false;
  }
}

function showMetrics() {
  const algo = state.algorithms.find((item) => item.slug === state.current);
  const detail = state.details[state.current];
  pushMessage(state.current, {
    role: "bot",
    embed: true,
    embedTitle: "Métriques actuelles",
    embedText: algo.how_to_read,
    fields: formatMetrics(detail?.metrics)
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        const [k, ...rest] = line.split(/\s{2,}|: /);
        return [k, rest.join(" ").trim() || "—"];
      }),
  });
}

function fillExample() {
  const algo = state.algorithms.find((item) => item.slug === state.current);
  fillFeatures(algo.sample_input || {});
}

function openCsvPicker() {
  els.csvInput.value = "";
  els.csvInput.click();
}

function downloadTemplate() {
  const link = document.createElement("a");
  link.href = `/algorithms/${state.current}/sample.csv`;
  link.download = `${state.current}-exemple.csv`;
  document.body.append(link);
  link.click();
  link.remove();
}

async function uploadCsv(file) {
  const algo = state.algorithms.find((item) => item.slug === state.current);
  if (!algo || state.busy) return;
  pushMessage(algo.slug, {
    role: "user",
    text: `Fichier CSV importé : ${file.name}`,
  });
  state.busy = true;
  els.sendBtn.disabled = true;
  els.uploadBtn.disabled = true;
  setTyping(true);
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("hyperparameters", JSON.stringify(collectHyperparameters(algo)));
    const response = await fetch(`/algorithms/${algo.slug}/predict-csv`, {
      method: "POST",
      body: form,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload.detail || response.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    setTyping(false);
    const fields = [
      ["Lignes", payload.row_count],
      ["Colonnes", (payload.columns_used || []).join(", ")],
    ];
    if (payload.match_rate != null) {
      fields.push(["Correspondances", `${payload.label_matches}/${payload.labeled_rows} (${payload.match_rate})`]);
    }
    pushMessage(algo.slug, {
      role: "bot",
      embed: true,
      embedTone: "success",
      embedTitle: "Prédictions CSV",
      embedText: payload.truncated
        ? "Seules les 50 premières lignes ont été lues."
        : `Fichier ${payload.filename}`,
      fields,
      tableHtml: csvTable(payload.rows),
    });
    updateSidebar(algo, payload);
  } catch (error) {
    setTyping(false);
    pushMessage(algo.slug, {
      role: "bot",
      embed: true,
      embedTone: "error",
      embedTitle: "CSV refusé",
      embedText: error.message,
    });
  } finally {
    state.busy = false;
    els.sendBtn.disabled = false;
    els.uploadBtn.disabled = false;
  }
}

els.modelTrigger.addEventListener("click", () => {
  const open = els.modelMenu.hidden;
  closeMenus();
  els.modelMenu.hidden = !open;
  els.modelTrigger.setAttribute("aria-expanded", String(open));
});

els.plusBtn.addEventListener("click", () => {
  const open = els.plusMenu.hidden;
  closeMenus();
  els.plusMenu.hidden = !open;
  els.plusBtn.setAttribute("aria-expanded", String(open));
});

els.plusMenu.addEventListener("click", (event) => {
  const action = event.target.closest("[data-action]")?.dataset.action;
  closeMenus();
  if (action === "example") fillExample();
  if (action === "exercise") runExercise();
  if (action === "upload") openCsvPicker();
  if (action === "template") downloadTemplate();
  if (action === "metrics") showMetrics();
});

els.messages.addEventListener("click", (event) => {
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "example") {
    fillExample();
    runPredict();
  }
  if (action === "exercise") runExercise();
  if (action === "upload") openCsvPicker();
});

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  runPredict();
});

els.uploadBtn.addEventListener("click", openCsvPicker);
els.csvInput.addEventListener("change", () => {
  const file = els.csvInput.files?.[0];
  if (file) uploadCsv(file);
});

document.addEventListener("click", (event) => {
  if (!event.target.closest(".model-picker") && !event.target.closest(".plus-btn") && !event.target.closest(".plus-menu")) {
    closeMenus();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeMenus();
    els.sidebar.classList.remove("open");
    els.backdrop.hidden = true;
  }
});

els.openSidebar.addEventListener("click", () => {
  els.sidebar.classList.add("open");
  els.backdrop.hidden = false;
});

els.closeSidebar.addEventListener("click", () => {
  els.sidebar.classList.remove("open");
  els.backdrop.hidden = true;
});

els.backdrop.addEventListener("click", () => {
  els.sidebar.classList.remove("open");
  els.backdrop.hidden = true;
});

async function boot() {
  try {
    const payload = await api("/algorithms");
    state.algorithms = payload.algorithms;
    els.apiDot.classList.add("ok");
    els.apiDot.parentElement.title = "API connectée";
    await selectAlgorithm(state.algorithms[0].slug);
  } catch (error) {
    els.modelLabel.textContent = "Hors ligne";
    els.messages.innerHTML = `<article class="message">
      ${pfpAvatar("tutor")}
      <div class="embed error"><h3>API injoignable</h3><p>${escapeHtml(error.message)}</p></div>
    </article>`;
  }
}

boot();
