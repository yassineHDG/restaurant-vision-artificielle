"use strict";

(function () {
  const RECONNECT_MS = 3000;
  const STATUS_MS = 1000;
  const HISTORY_MS = 10000;
  const stream = document.getElementById("stream");
  const videoFrame = document.querySelector(".video-frame");
  const fullscreenButton = document.getElementById("video-fullscreen");
  const entranceStream = document.getElementById("entrance-stream");
  const entrancePlaceholder = document.getElementById("entrance-placeholder");
  const overlay = document.getElementById("video-overlay");
  const overlayText = document.getElementById("video-overlay-text");
  const connectionDot = document.getElementById("connection-dot");
  const connectionLabel = document.getElementById("connection-label");
  const processingBadge = document.getElementById("processing-badge");
  const tableList = document.getElementById("table-list");
  const settingsDialog = document.getElementById("settings-dialog");
  const settingsForm = document.getElementById("settings-form");
  let reconnectTimer = null;
  let entranceReconnectTimer = null;
  let processingState = "starting";
  let entranceActive = false;
  let entranceConnected = false;
  let lastStatus = {tables: [], cameras: {}};
  let operationsSnapshot = {tables: {}, alerts: [], cameras: {}};
  let historySnapshot = null;

  function setText(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
  }

  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) return "—";
    const total = Math.max(0, Math.floor(seconds));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return hours > 0
      ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
      : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  function formatWait(seconds) {
    if (seconds === null || seconds === undefined) return "—";
    if (seconds < 60) return `${Math.round(seconds)} s`;
    return `${Math.round(seconds / 60)} min`;
  }

  function setConnection(state, label) {
    connectionDot.className = `status-dot${state ? ` is-${state}` : ""}`;
    connectionLabel.textContent = label;
  }

  function setFullscreenState(active) {
    fullscreenButton.classList.toggle("is-active", active);
    fullscreenButton.setAttribute("aria-label", active ? "Quitter le plein écran" : "Afficher la vidéo en plein écran");
    fullscreenButton.lastElementChild.textContent = active ? "Quitter" : "Plein écran HD";
  }

  function enableFullscreenFallback() {
    videoFrame.classList.add("is-fullscreen-fallback");
    document.body.classList.add("has-video-fullscreen");
    setFullscreenState(true);
  }

  function disableFullscreenFallback() {
    videoFrame.classList.remove("is-fullscreen-fallback");
    document.body.classList.remove("has-video-fullscreen");
    setFullscreenState(false);
  }

  async function toggleVideoFullscreen() {
    if (videoFrame.classList.contains("is-fullscreen-fallback")) return disableFullscreenFallback();
    if (document.fullscreenElement === videoFrame) return document.exitFullscreen();
    if (videoFrame.requestFullscreen) {
      try { await videoFrame.requestFullscreen(); } catch (_error) { enableFullscreenFallback(); }
    } else enableFullscreenFallback();
  }

  const stateLabels = {free: "Libre", occupied: "Occupée", needs_cleaning: "À nettoyer", ready: "Prête", available: "Libre"};

  async function runTableAction(tableId, action, button) {
    button.disabled = true;
    try {
      const response = await fetch(`/api/tables/${encodeURIComponent(tableId)}/actions/${action}`, {method: "POST"});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Action refusée");
      await pollOperations();
    } catch (error) {
      button.disabled = false;
      button.title = error.message;
    }
  }

  function renderTables(tables) {
    tableList.replaceChildren();
    if (!tables.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "Aucune table configurée.";
      tableList.appendChild(empty);
      return;
    }
    for (const table of tables) {
      const operational = (operationsSnapshot.tables || {})[table.id] || {};
      const tableState = operational.state || table.state || table.status || "free";
      const row = document.createElement("div");
      row.className = `table-row table-row-${tableState}`;
      const name = document.createElement("strong");
      name.textContent = table.id;
      const guests = document.createElement("span");
      guests.className = "numeric";
      guests.textContent = `${table.guests}/${table.capacity}`;
      const duration = document.createElement("span");
      duration.className = "numeric";
      duration.textContent = formatDuration(table.occupancy_seconds);
      const release = document.createElement("span");
      release.className = "numeric release-time";
      release.textContent = operational.predicted_release_seconds === null || operational.predicted_release_seconds === undefined ? "—" : formatWait(operational.predicted_release_seconds);
      const stateCell = document.createElement("div");
      stateCell.className = "table-state-cell";
      const state = document.createElement("span");
      state.className = `state state-${tableState}`;
      state.textContent = stateLabels[tableState] || tableState;
      stateCell.appendChild(state);
      const action = document.createElement("button");
      action.type = "button";
      action.className = "table-action";
      if (tableState === "occupied" && !operational.served) {
        action.textContent = "Marquer servie";
        action.addEventListener("click", () => runTableAction(table.id, "served", action));
        stateCell.appendChild(action);
      } else if (tableState === "occupied" && operational.served) {
        action.textContent = "Service confirmé";
        action.disabled = true;
        stateCell.appendChild(action);
      } else if (tableState === "needs_cleaning") {
        action.textContent = "Marquer prête";
        action.addEventListener("click", () => runTableAction(table.id, "ready", action));
        stateCell.appendChild(action);
      }
      row.append(name, guests, duration, release, stateCell);
      tableList.appendChild(row);
    }
  }

  function renderStatus(data) {
    lastStatus = data;
    processingState = data.state;
    setText("metric-guests", data.seated_guests);
    setText("metric-capacity", `sur ${data.total_capacity} places`);
    const rate = data.occupancy_rate !== undefined ? data.occupancy_rate : (data.total_tables ? data.occupied_tables / data.total_tables * 100 : 0);
    setText("metric-occupancy-rate", `${Number(rate).toFixed(0)}%`);
    setText("metric-tables-detail", `${data.occupied_tables} occupée${data.occupied_tables > 1 ? "s" : ""} · ${data.available_tables} disponible${data.available_tables > 1 ? "s" : ""}`);
    setText("source-label", data.source || "Source vidéo");
    setText("table-count-badge", `${data.total_tables} table${data.total_tables > 1 ? "s" : ""}`);
    setText("video-time", formatDuration(data.video_time_seconds));
    renderTables(data.tables || []);
    const gpu = data.gpu || {};
    setText("gpu-label", gpu.name ? `${gpu.name} · ${gpu.memory_reserved_mib || 0} MiB` : "GPU non détecté");
    entranceActive = Boolean(data.cameras && data.cameras.entrance && data.cameras.entrance.active);
    if (entranceActive && !entranceConnected) {
      entranceConnected = true;
      entrancePlaceholder.hidden = true;
      connectEntranceStream();
    } else if (!entranceActive) {
      entrancePlaceholder.hidden = false;
      setText("queue-badge", "Non configurée");
    }
    const labels = {starting: "Initialisation", warming_up: "Préparation GPU", running: "Analyse en direct", completed: "Vidéo terminée", stopped: "Traitement arrêté", reconnecting: "Reconnexion caméra", stale: "Flux interrompu", error: "Erreur du modèle", idle: "En attente"};
    processingBadge.textContent = labels[data.state] || data.state;
    processingBadge.classList.toggle("is-error", data.state === "error");
    if (["completed", "stopped"].includes(data.state)) {
      overlay.hidden = true;
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    }
    if (data.state === "error") {
      setConnection("error", "Erreur de traitement");
      overlay.hidden = false;
      overlayText.textContent = data.error || "Le traitement vidéo a échoué.";
    }
  }

  function renderQueue(data) {
    const confirmed = data.confirmed_queue_count || 0;
    const uncertain = data.uncertain_queue_count || 0;
    const groups = data.confirmed_group_count || 0;
    setText("metric-queue", confirmed);
    setText("metric-queue-detail", `${groups} groupe${groups > 1 ? "s" : ""} · ${uncertain} incertain${uncertain > 1 ? "s" : ""}`);
    if (!operationsSnapshot.average_wait_seconds) setText("metric-average-wait", formatWait(data.average_wait_seconds || 0));
    setText("queue-confirmed", confirmed);
    setText("queue-uncertain", uncertain);
    setText("queue-staff", data.staff_visible || 0);
    setText("queue-badge", data.calibration_required ? "À calibrer" : (data.state === "running" ? "Analyse en direct" : "En attente"));
    setText("queue-summary", data.calibration_required ? "Dessinez la zone de file avant de compter les attentes." : `${confirmed} personne${confirmed > 1 ? "s" : ""} dans ${groups} groupe${groups > 1 ? "s" : ""}. Attente moyenne ${formatWait(data.average_wait_seconds || 0)}.`);
    const container = document.getElementById("queue-tracks");
    container.replaceChildren();
    for (const track of (data.tracks || []).filter(item => item.queue_state !== "outside").slice(0, 8)) {
      const row = document.createElement("div");
      row.className = "queue-track";
      const label = document.createElement("span");
      label.textContent = `#${track.track_id}${track.group_id ? ` · G${track.group_id}` : ""} · ${track.role}`;
      const duration = document.createElement("strong");
      duration.textContent = formatDuration(track.dwell_seconds);
      row.append(label, duration);
      container.appendChild(row);
    }
  }

  function renderAlerts(alerts) {
    const list = document.getElementById("alerts-list");
    list.replaceChildren();
    setText("alert-count", `${alerts.length} active${alerts.length > 1 ? "s" : ""}`);
    if (!alerts.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "Aucune alerte active.";
      list.appendChild(empty);
      return;
    }
    for (const alert of alerts) {
      const item = document.createElement("article");
      item.className = `alert-item alert-${alert.severity}${alert.acknowledged ? " is-acknowledged" : ""}`;
      const content = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = alert.title;
      const message = document.createElement("p");
      message.textContent = alert.message;
      content.append(title, message);
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = alert.acknowledged ? "Acquittée" : "Acquitter";
      button.disabled = Boolean(alert.acknowledged);
      button.addEventListener("click", async () => {
        button.disabled = true;
        await fetch(`/api/alerts/${encodeURIComponent(alert.id)}/acknowledge`, {method: "POST"});
        await pollOperations();
      });
      item.append(content, button);
      list.appendChild(item);
    }
  }

  function renderCameraHealth(cameras) {
    const list = document.getElementById("camera-health-list");
    list.replaceChildren();
    const values = Object.values(cameras || {});
    const unhealthy = values.some(camera => ["error", "stale", "reconnecting"].includes(camera.state));
    setText("camera-health-badge", unhealthy ? "Attention" : "Flux stables");
    document.getElementById("camera-health-badge").classList.toggle("is-error", unhealthy);
    for (const camera of values) {
      const row = document.createElement("div");
      row.className = "camera-health-row";
      const name = document.createElement("span");
      name.textContent = camera.label || camera.role;
      const state = document.createElement("strong");
      state.className = `camera-state camera-state-${camera.state}`;
      state.textContent = camera.active ? `${camera.state}${camera.reconnects ? ` · ${camera.reconnects} reconnexion${camera.reconnects > 1 ? "s" : ""}` : ""}` : "non configurée";
      row.append(name, state);
      list.appendChild(row);
    }
  }

  function renderOperations(data) {
    operationsSnapshot = data;
    setText("metric-occupancy-rate", `${Number(data.occupancy_rate || 0).toFixed(0)}%`);
    setText("metric-average-wait", formatWait(data.average_wait_seconds || 0));
    setText("metric-staff", data.staff_visible || 0);
    setText("metric-estimated-wait", formatWait(data.estimated_wait_seconds || 0));
    setText("queue-estimated-wait", formatWait(data.estimated_wait_seconds || 0));
    setText("ops-queue-groups", data.queue_groups || 0);
    const tables = Object.values(data.tables || {});
    setText("ops-ready-tables", tables.filter(table => ["free", "ready"].includes(table.state)).length);
    setText("ops-cleaning-tables", tables.filter(table => table.state === "needs_cleaning").length);
    const releases = tables.map(table => table.predicted_release_seconds).filter(value => value !== null && value !== undefined);
    setText("ops-next-release", releases.length ? formatWait(Math.min(...releases)) : "—");
    renderAlerts(data.alerts || []);
    renderCameraHealth(data.cameras || lastStatus.cameras || {});
    renderTables(lastStatus.tables || []);
  }

  async function pollStatus() {
    try {
      const response = await fetch("/api/status", {cache: "no-store"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      renderStatus(await response.json());
      await Promise.all([pollQueue(), pollOperations()]);
      setConnection("live", "Backend connecté");
    } catch (_error) { setConnection("error", "Backend hors ligne"); }
  }

  async function pollQueue() {
    try { const response = await fetch("/api/queue", {cache: "no-store"}); if (response.ok) renderQueue(await response.json()); } catch (_error) { /* status owns offline state */ }
  }

  async function pollOperations() {
    try { const response = await fetch("/api/operations", {cache: "no-store"}); if (response.ok) renderOperations(await response.json()); } catch (_error) { /* status owns offline state */ }
  }

  function connectStream() { stream.src = `/stream/dining?t=${Date.now()}`; }
  function connectEntranceStream() { entranceStream.src = `/stream/entrance?t=${Date.now()}`; }

  stream.addEventListener("load", () => { overlay.hidden = true; });
  stream.addEventListener("error", () => {
    if (["completed", "stopped"].includes(processingState) || reconnectTimer !== null) return;
    overlay.hidden = false;
    overlayText.textContent = "Flux vidéo indisponible — nouvelle tentative…";
    reconnectTimer = setTimeout(() => { reconnectTimer = null; connectStream(); }, RECONNECT_MS);
  });
  entranceStream.addEventListener("load", () => { entrancePlaceholder.hidden = true; });
  entranceStream.addEventListener("error", () => {
    if (!entranceActive || processingState === "completed" || entranceReconnectTimer !== null) return;
    entrancePlaceholder.hidden = false;
    entrancePlaceholder.textContent = "Flux d’entrée indisponible — nouvelle tentative…";
    entranceReconnectTimer = setTimeout(() => { entranceReconnectTimer = null; connectEntranceStream(); }, RECONNECT_MS);
  });
  fullscreenButton.addEventListener("click", toggleVideoFullscreen);
  stream.addEventListener("dblclick", toggleVideoFullscreen);
  document.addEventListener("fullscreenchange", () => setFullscreenState(document.fullscreenElement === videoFrame));
  document.addEventListener("keydown", event => { if (event.key === "Escape" && videoFrame.classList.contains("is-fullscreen-fallback")) disableFullscreenFallback(); });

  function drawHistory(data) {
    const canvas = document.getElementById("history-chart");
    const empty = document.getElementById("history-empty");
    const points = data.hourly || [];
    empty.hidden = points.length > 0;
    canvas.hidden = points.length === 0;
    if (!points.length) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(420, Math.floor(rect.width || 760));
    const height = 280;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const pad = {left: 42, right: 18, top: 18, bottom: 34};
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;
    context.font = "11px Inter, system-ui, sans-serif";
    context.fillStyle = "#667579";
    context.strokeStyle = "#dfe6e5";
    context.lineWidth = 1;
    for (let value = 0; value <= 100; value += 25) {
      const y = pad.top + plotHeight - value / 100 * plotHeight;
      context.beginPath(); context.moveTo(pad.left, y); context.lineTo(width - pad.right, y); context.stroke();
      context.fillText(`${value}%`, 4, y + 4);
    }
    context.strokeStyle = "#0f766e";
    context.lineWidth = 2.5;
    context.beginPath();
    points.forEach((point, index) => {
      const x = pad.left + (points.length === 1 ? plotWidth / 2 : index / (points.length - 1) * plotWidth);
      const y = pad.top + plotHeight - Math.max(0, Math.min(100, point.occupancy_rate)) / 100 * plotHeight;
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
    context.fillStyle = "#152024";
    const labelIndexes = new Set([0, Math.floor((points.length - 1) / 2), points.length - 1]);
    points.forEach((point, index) => {
      if (!labelIndexes.has(index)) return;
      const x = pad.left + (points.length === 1 ? plotWidth / 2 : index / (points.length - 1) * plotWidth);
      const label = point.bucket ? (point.bucket.slice(11, 16) || point.bucket.slice(5)) : "";
      context.fillText(label, Math.max(pad.left, x - 16), height - 10);
    });
  }

  function renderHistory(data) {
    historySnapshot = data;
    drawHistory(data);
    const list = document.getElementById("turnover-list");
    list.replaceChildren();
    for (const table of (data.turnover || [])) {
      const row = document.createElement("div");
      row.className = "turnover-row";
      const name = document.createElement("strong"); name.textContent = table.table_id;
      const groups = document.createElement("span"); groups.textContent = `${table.groups_served} groupe${table.groups_served > 1 ? "s" : ""}`;
      const duration = document.createElement("span"); duration.textContent = table.average_visit_seconds ? `moy. ${formatWait(table.average_visit_seconds)}` : "pas encore de durée";
      row.append(name, groups, duration); list.appendChild(row);
    }
    if (!(data.turnover || []).length) {
      const empty = document.createElement("p"); empty.className = "empty-state"; empty.textContent = "Aucune rotation enregistrée."; list.appendChild(empty);
    }
    const peaks = document.getElementById("peak-list");
    peaks.replaceChildren();
    for (const peak of (data.peak_hours || [])) {
      const row = document.createElement("div"); row.className = "peak-row";
      const time = document.createElement("strong"); time.textContent = peak.bucket?.slice(11, 16) || peak.bucket;
      const occupancy = document.createElement("span"); occupancy.textContent = `pic ${Number(peak.peak_occupancy_rate).toFixed(0)}%`;
      const staff = document.createElement("span"); staff.textContent = `${peak.recommended_staff} staff conseillé${peak.recommended_staff > 1 ? "s" : ""}`;
      row.append(time, occupancy, staff); peaks.appendChild(row);
    }
    if (!(data.peak_hours || []).length) {
      const empty = document.createElement("p"); empty.className = "empty-state compact"; empty.textContent = "Pas encore de pic mesuré."; peaks.appendChild(empty);
    }
  }

  async function pollHistory() {
    const hours = Number(document.getElementById("history-range").value || 24);
    document.getElementById("export-stats").href = `/api/operations/export.csv?hours=${hours}`;
    try { const response = await fetch(`/api/operations/history?hours=${hours}`, {cache: "no-store"}); if (response.ok) renderHistory(await response.json()); } catch (_error) { /* live view remains available */ }
  }

  async function openSettings() {
    const message = document.getElementById("settings-message");
    message.textContent = "Chargement…";
    settingsDialog.showModal();
    try {
      const response = await fetch("/api/operations/settings", {cache: "no-store"});
      const saved = await response.json();
      for (const [key, value] of Object.entries(saved)) {
        const field = settingsForm.elements.namedItem(key);
        if (field) field.value = value;
      }
      message.textContent = "";
    } catch (_error) { message.textContent = "Impossible de charger les seuils."; }
  }

  settingsForm.addEventListener("submit", async event => {
    event.preventDefault();
    const payload = {};
    for (const name of ["queue_alert_people", "max_queue_wait_seconds", "service_alert_seconds", "cleaning_alert_seconds", "default_dining_minutes", "queue_group_join_seconds", "camera_reconnect_attempts", "stream_jpeg_quality"]) {
      payload[name] = Number(settingsForm.elements.namedItem(name).value);
    }
    const message = document.getElementById("settings-message");
    message.textContent = "Enregistrement…";
    try {
      const response = await fetch("/api/operations/settings", {method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Paramètres invalides");
      message.textContent = result.restart_required ? "Seuils enregistrés. Redémarrez le backend pour la qualité vidéo ou la reconnexion caméra." : "Seuils appliqués immédiatement.";
      await pollOperations();
    } catch (error) { message.textContent = error.message; }
  });

  document.getElementById("settings-button").addEventListener("click", openSettings);
  document.getElementById("settings-close").addEventListener("click", () => settingsDialog.close());
  document.getElementById("settings-cancel").addEventListener("click", () => settingsDialog.close());
  document.getElementById("history-range").addEventListener("change", pollHistory);
  window.addEventListener("resize", () => { if (historySnapshot) drawHistory(historySnapshot); });

  function updateClock() {
    const now = new Date();
    const clock = document.getElementById("live-clock");
    clock.dateTime = now.toISOString();
    clock.textContent = new Intl.DateTimeFormat("fr-FR", {hour: "2-digit", minute: "2-digit", second: "2-digit"}).format(now);
  }

  connectStream();
  pollStatus();
  pollHistory();
  updateClock();
  setInterval(pollStatus, STATUS_MS);
  setInterval(pollHistory, HISTORY_MS);
  setInterval(updateClock, 1000);
})();
