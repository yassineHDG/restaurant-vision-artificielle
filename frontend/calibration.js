"use strict";

(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const overlay = document.getElementById("zone-overlay");
  const cameraSelect = document.getElementById("camera-select");
  const tableSelect = document.getElementById("table-select");
  const tableCapacity = document.getElementById("table-capacity");
  const stream = document.getElementById("calibration-stream");
  const drawingStage = document.getElementById("drawing-stage");
  const message = document.getElementById("save-message");
  let config = null;
  let activeZone = "queue_polygon";
  let activeTableZone = "display_polygon";
  let samplingStaff = false;
  let operationalTableStates = {};

  function currentCamera() {
    return config.cameras.find(camera => camera.id === cameraSelect.value);
  }

  function isDining() { return cameraSelect.value === "dining"; }

  function ensureRegions(camera) {
    camera.regions = camera.regions || {};
    camera.regions.tables = camera.regions.tables || [];
    for (const key of ["queue_polygon", "transit_polygon", "interior_polygon", "entry_line"]) {
      camera.regions[key] = camera.regions[key] || [];
    }
  }

  function ensureStaffMarker(camera) {
    camera.staff_marker = camera.staff_marker || {
      hsv_lower: [80, 80, 60],
      hsv_upper: [110, 255, 255],
      minimum_ratio: 0.025
    };
    return camera.staff_marker;
  }

  function currentTable() {
    const tables = currentCamera().regions.tables;
    return tables.find(table => table.id === tableSelect.value) || tables[0] || null;
  }

  function nextTableId() {
    const ids = new Set(currentCamera().regions.tables.map(table => table.id));
    let number = 1;
    while (ids.has(`T-${String(number).padStart(2, "0")}`)) number += 1;
    return `T-${String(number).padStart(2, "0")}`;
  }

  function ensureTable() {
    let table = currentTable();
    if (!table) {
      table = {id: nextTableId(), capacity: 4, counting_polygon: [], display_polygon: []};
      currentCamera().regions.tables.push(table);
    }
    return table;
  }

  function activePoints() {
    if (isDining()) return ensureTable()[activeTableZone];
    return currentCamera().regions[activeZone];
  }

  function expandPolygon(points, factor = 1.65) {
    if (!Array.isArray(points) || points.length < 3) return [];
    const center = points.reduce((sum, point) => [sum[0] + point[0], sum[1] + point[1]], [0, 0])
      .map(value => value / points.length);
    return points.map(([x, y]) => [
      Number(Math.max(0, Math.min(1, center[0] + (x - center[0]) * factor)).toFixed(6)),
      Number(Math.max(0, Math.min(1, center[1] + (y - center[1]) * factor)).toFixed(6))
    ]);
  }

  function selectTableZone(zone) {
    activeTableZone = zone;
    document.querySelectorAll(".table-zone-button").forEach(button => {
      button.classList.toggle("is-active", button.dataset.tableZone === zone);
    });
  }

  function renderStaffControls() {
    const marker = ensureStaffMarker(currentCamera());
    document.getElementById("staff-h-min").value = marker.hsv_lower[0];
    document.getElementById("staff-h-max").value = marker.hsv_upper[0];
    document.getElementById("staff-s-min").value = marker.hsv_lower[1];
    document.getElementById("staff-v-min").value = marker.hsv_lower[2];
    document.getElementById("staff-min-ratio").value = Number((marker.minimum_ratio * 100).toFixed(2));
    document.getElementById("sample-staff-color").classList.toggle("is-active", samplingStaff);
  }

  function updateStaffMarker() {
    const marker = ensureStaffMarker(currentCamera());
    const number = (id, fallback) => {
      const value = Number.parseFloat(document.getElementById(id).value);
      return Number.isFinite(value) ? value : fallback;
    };
    marker.hsv_lower = [
      Math.max(0, Math.min(179, number("staff-h-min", marker.hsv_lower[0]))),
      Math.max(0, Math.min(255, number("staff-s-min", marker.hsv_lower[1]))),
      Math.max(0, Math.min(255, number("staff-v-min", marker.hsv_lower[2])))
    ].map(Math.round);
    marker.hsv_upper = [
      Math.max(marker.hsv_lower[0], Math.min(179, number("staff-h-max", marker.hsv_upper[0]))),
      255,
      255
    ].map(Math.round);
    marker.minimum_ratio = Math.max(0.001, Math.min(0.5, number("staff-min-ratio", marker.minimum_ratio * 100) / 100));
  }

  async function sampleStaffColor(point) {
    try {
      const response = await fetch(
        `/api/calibration/color-sample/${cameraSelect.value}?x=${point[0]}&y=${point[1]}`,
        {cache: "no-store"}
      );
      const sample = await response.json();
      if (!response.ok) throw new Error(sample.detail || "Échantillon indisponible");
      const [h, s, v] = sample.hsv;
      const marker = ensureStaffMarker(currentCamera());
      marker.hsv_lower = [Math.max(0, h - 10), Math.max(40, s - 60), Math.max(40, v - 80)];
      marker.hsv_upper = [Math.min(179, h + 10), 255, 255];
      const [b, g, r] = sample.bgr;
      document.getElementById("staff-swatch").style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
      samplingStaff = false;
      message.textContent = `Couleur staff prélevée (HSV ${h}, ${s}, ${v}). Vérifiez la tolérance puis enregistrez.`;
      renderStaffControls();
    } catch (error) {
      samplingStaff = false;
      message.textContent = error.message;
      renderStaffControls();
    }
  }

  function makeSvg(name, attributes) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  }

  function drawShape(points, className, line = false, showPoints = true, attributes = {}) {
    if (!Array.isArray(points) || !points.length) return;
    const scaled = points.map(([x, y]) => `${x * 1000},${y * 562}`).join(" ");
    overlay.appendChild(makeSvg(line ? "polyline" : "polygon", {
      points: scaled,
      class: `zone-shape ${className}`,
      ...attributes
    }));
    if (showPoints) for (const [x, y] of points) overlay.appendChild(makeSvg("circle", {
      cx: x * 1000,
      cy: y * 562,
      r: 6,
      class: "zone-point"
    }));
  }

  function applyOperationalTableStyles() {
    const stateClasses = [
      "table-state-free",
      "table-state-ready",
      "table-state-occupied",
      "table-state-needs_cleaning"
    ];
    overlay.querySelectorAll("[data-table-id]").forEach(shape => {
      shape.classList.remove(...stateClasses);
      const state = operationalTableStates[shape.dataset.tableId]?.state || "free";
      shape.classList.add(`table-state-${state}`);
    });
  }

  function renderTableControls() {
    const dining = isDining();
    document.getElementById("entrance-controls").hidden = dining;
    document.getElementById("dining-controls").hidden = !dining;
    if (!dining) return;
    const previous = tableSelect.value;
    tableSelect.replaceChildren();
    for (const table of currentCamera().regions.tables) {
      const option = document.createElement("option");
      option.value = table.id;
      option.textContent = `${table.id} — ${table.capacity} places`;
      tableSelect.appendChild(option);
    }
    const table = ensureTable();
    tableSelect.value = currentCamera().regions.tables.some(item => item.id === previous)
      ? previous
      : table.id;
    tableCapacity.value = currentTable().capacity;
  }

  function render() {
    overlay.replaceChildren();
    const camera = currentCamera();
    ensureRegions(camera);
    document.getElementById("camera-title").textContent = camera.label || camera.id;
    if (camera.calibration_required) {
      message.textContent = "La source vidéo a changé. Les anciennes zones sont affichées comme référence mais restent désactivées dans l’analyse jusqu’à l’enregistrement d’une nouvelle calibration.";
    }
    renderStaffControls();
    renderTableControls();
    if (isDining()) {
      const selected = currentTable();
      for (const table of camera.regions.tables) {
        const operationalState = operationalTableStates[table.id]?.state || "free";
        drawShape(
          table.counting_polygon,
          `table_counting table-state-${operationalState}${table === selected ? " is-selected" : ""}`,
          false,
          table === selected && activeTableZone === "counting_polygon",
          {"data-table-id": table.id}
        );
        drawShape(
          table.display_polygon,
          `table_display table-state-${operationalState}${table === selected ? " is-selected" : ""}`,
          false,
          table === selected && activeTableZone === "display_polygon",
          {"data-table-id": table.id}
        );
      }
    } else {
      for (const key of ["queue_polygon", "transit_polygon", "interior_polygon", "entry_line"]) {
        drawShape(camera.regions[key], key, key === "entry_line");
      }
    }
  }

  overlay.addEventListener("click", async event => {
    const bounds = overlay.getBoundingClientRect();
    const point = [
      Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height))
    ].map(value => Number(value.toFixed(6)));
    if (samplingStaff) {
      await sampleStaffColor(point);
      return;
    }
    if (!isDining() && activeZone === "entry_line" && activePoints().length >= 2) {
      currentCamera().regions[activeZone] = [];
    }
    activePoints().push(point);
    render();
  });

  for (const button of document.querySelectorAll(".zone-button")) {
    button.addEventListener("click", () => {
      activeZone = button.dataset.zone;
      document.querySelectorAll(".zone-button").forEach(item => item.classList.toggle("is-active", item === button));
    });
  }

  for (const button of document.querySelectorAll(".table-zone-button")) {
    button.addEventListener("click", () => selectTableZone(button.dataset.tableZone));
  }

  document.getElementById("sample-staff-color").addEventListener("click", () => {
    samplingStaff = !samplingStaff;
    message.textContent = samplingStaff
      ? "Cliquez maintenant sur la couleur du tablier dans l’image."
      : "Prélèvement annulé.";
    renderStaffControls();
  });
  for (const id of ["staff-h-min", "staff-h-max", "staff-s-min", "staff-v-min", "staff-min-ratio"]) {
    document.getElementById(id).addEventListener("change", updateStaffMarker);
  }

  document.getElementById("undo-point").addEventListener("click", () => {
    activePoints().pop();
    render();
  });
  document.getElementById("clear-zone").addEventListener("click", () => {
    if (isDining()) {
      ensureTable()[activeTableZone] = [];
    } else currentCamera().regions[activeZone] = [];
    render();
  });
  document.getElementById("expand-table-zone").addEventListener("click", () => {
    const table = ensureTable();
    if (table.display_polygon.length < 3) {
      message.textContent = "Dessinez d’abord la surface de la table.";
      return;
    }
    table.counting_polygon = expandPolygon(table.display_polygon);
    selectTableZone("counting_polygon");
    message.textContent = "Zone clients générée. Ajustez-la pour englober toutes les chaises.";
    render();
  });
  document.getElementById("new-table").addEventListener("click", () => {
    const table = {id: nextTableId(), capacity: 4, counting_polygon: [], display_polygon: []};
    currentCamera().regions.tables.push(table);
    selectTableZone("display_polygon");
    render();
    tableSelect.value = table.id;
    render();
  });
  document.getElementById("delete-table").addEventListener("click", () => {
    const tables = currentCamera().regions.tables;
    const table = currentTable();
    if (!table) return;
    if (tables.length <= 1) {
      message.textContent = "La salle doit conserver au moins une table.";
      return;
    }
    currentCamera().regions.tables = tables.filter(item => item.id !== table.id);
    message.textContent = `${table.id} retirée localement. Cliquez sur « Enregistrer la calibration » pour confirmer.`;
    render();
  });
  tableSelect.addEventListener("change", render);
  function updateTableCapacity() {
    ensureTable().capacity = Math.max(1, Number.parseInt(tableCapacity.value, 10) || 1);
    render();
  }
  tableCapacity.addEventListener("change", updateTableCapacity);
  document.getElementById("load-proposals").addEventListener("click", async () => {
    try {
      const response = await fetch("/api/calibration/table-proposals", {cache: "no-store"});
      const draft = await response.json();
      if (!response.ok) throw new Error(draft.detail || "Brouillon illisible");
      if (!draft.tables || !draft.tables.length) {
        throw new Error("Aucune proposition. Exécutez d’abord la commande YOLO Seg.");
      }
      const previousCapacities = new Map(
        currentCamera().regions.tables.map(table => [table.id, table.capacity])
      );
      currentCamera().regions.tables = draft.tables.map((proposal, index) => {
        const id = `T-${String(index + 1).padStart(2, "0")}`;
        return {
          id,
          capacity: previousCapacities.get(id) || 4,
          counting_polygon: expandPolygon(proposal.polygon),
          display_polygon: proposal.polygon
        };
      });
      message.textContent = `${draft.tables.length} proposition(s) chargée(s). Vérifiez les polygones et capacités avant d’enregistrer.`;
      render();
    } catch (error) { message.textContent = error.message; }
  });
  cameraSelect.addEventListener("change", () => {
    samplingStaff = false;
    stream.src = `/api/calibration/frame/${cameraSelect.value}?t=${Date.now()}`;
    render();
  });
  stream.addEventListener("load", () => {
    if (stream.naturalWidth && stream.naturalHeight) {
      drawingStage.style.aspectRatio = `${stream.naturalWidth} / ${stream.naturalHeight}`;
    }
  });
  document.getElementById("save-calibration").addEventListener("click", async () => {
    try {
      if (isDining()) updateTableCapacity();
      updateStaffMarker();
      const response = await fetch("/api/calibration", {
        method: "PUT",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(config)
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Configuration invalide");
      config = result.config;
      document.getElementById("save-state").textContent = "Enregistré";
      message.textContent = "Calibration enregistrée. Redémarrez le backend pour l’appliquer.";
    } catch (error) { message.textContent = error.message; }
  });

  fetch("/api/calibration", {cache: "no-store"}).then(response => response.json()).then(data => {
    config = data;
    config.cameras.forEach(camera => { ensureRegions(camera); ensureStaffMarker(camera); });
    render();
  });

  async function refreshOperationalStates() {
    try {
      const response = await fetch("/api/operations", {cache: "no-store"});
      if (!response.ok) return;
      const operations = await response.json();
      operationalTableStates = operations.tables || {};
      if (config && isDining()) applyOperationalTableStyles();
    } catch (_error) { /* Calibration remains usable while the backend recovers. */ }
  }
  refreshOperationalStates();
  setInterval(refreshOperationalStates, 2000);
})();
