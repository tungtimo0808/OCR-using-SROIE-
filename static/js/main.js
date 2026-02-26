/* =========================================================
   NEURAL OCR – Frontend Logic
   ========================================================= */

(function () {
  "use strict";

  // ── Particles ──────────────────────────────────────────
  const particleContainer = document.getElementById("particles");
  for (let i = 0; i < 30; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    p.style.cssText = `
      left: ${Math.random() * 100}%;
      --dur: ${6 + Math.random() * 10}s;
      --delay: ${-Math.random() * 12}s;
      --dx: ${(Math.random() - 0.5) * 80}px;
      opacity: ${0.3 + Math.random() * 0.4};
      width: ${Math.random() > 0.7 ? 3 : 2}px;
      height: ${Math.random() > 0.7 ? 3 : 2}px;
    `;
    particleContainer.appendChild(p);
  }

  // ── Element refs ───────────────────────────────────────
  const dropZone    = document.getElementById("dropZone");
  const fileInput   = document.getElementById("fileInput");
  const dropContent = document.getElementById("dropContent");
  const previewInner= document.getElementById("previewInner");
  const previewImg  = document.getElementById("previewImg");
  const clearBtn    = document.getElementById("clearBtn");
  const browseBtn   = document.getElementById("browseBtn");
  const fileInfo    = document.getElementById("fileInfo");
  const fiName      = document.getElementById("fiName");
  const fiSize      = document.getElementById("fiSize");
  const fiType      = document.getElementById("fiType");
  const runBtn      = document.getElementById("runBtn");
  const copyBtn     = document.getElementById("copyBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  const retryBtn    = document.getElementById("retryBtn");

  const idleState    = document.getElementById("idleState");
  const loadingState = document.getElementById("loadingState");
  const resultState  = document.getElementById("resultState");
  const errorState   = document.getElementById("errorState");
  const errorMsg     = document.getElementById("errorMsg");
  const resultActions= document.getElementById("resultActions");
  const loaderText   = document.getElementById("loaderText");

  const fCompany = document.getElementById("fCompany");
  const fDate    = document.getElementById("fDate");
  const fAddress = document.getElementById("fAddress");
  const fTotal   = document.getElementById("fTotal");
  const dBoxes   = document.getElementById("dBoxes");
  const dWords   = document.getElementById("dWords");
  const dOcr     = document.getElementById("dOcr");
  const toast    = document.getElementById("toast");

  let currentFile = null;
  let lastFields  = null;

  // ── Utilities ──────────────────────────────────────────
  function showToast(msg, dur = 2200) {
    toast.textContent = msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), dur);
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 ** 2) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1024 ** 2).toFixed(1) + " MB";
  }

  function showState(name) {
    idleState.style.display    = name === "idle"    ? "flex" : "none";
    loadingState.style.display = name === "loading" ? "flex" : "none";
    resultState.style.display  = name === "result"  ? "block" : "none";
    errorState.style.display   = name === "error"   ? "flex" : "none";
    resultActions.style.display= name === "result"  ? "flex" : "none";
  }

  // ── Drop zone ──────────────────────────────────────────
  browseBtn.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("click", (e) => {
    if (e.target === clearBtn || clearBtn.contains(e.target)) return;
    if (!previewInner.style.display || previewInner.style.display === "none") {
      fileInput.click();
    }
  });

  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) loadFile(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) loadFile(fileInput.files[0]);
  });

  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
  });

  function loadFile(file) {
    const allowed = ["image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"];
    if (!allowed.some(t => file.type.startsWith(t.split("/")[0]) && file.name.match(/\.(png|jpg|jpeg|bmp|tiff|webp)$/i))) {
      showToast("⚠ Unsupported file type");
      return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (ev) => {
      previewImg.src = ev.target.result;
      dropContent.style.display = "none";
      previewInner.style.display = "flex";
    };
    reader.readAsDataURL(file);

    fiName.textContent = file.name.length > 24 ? file.name.slice(0, 22) + "…" : file.name;
    fiSize.textContent = formatBytes(file.size);
    fiType.textContent = file.type || "image/*";
    fileInfo.style.display = "flex";
    runBtn.disabled = false;
    showState("idle");
  }

  function clearFile() {
    currentFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewInner.style.display = "none";
    dropContent.style.display = "flex";
    fileInfo.style.display = "none";
    runBtn.disabled = true;
    showState("idle");
  }

  // ── Step animation ─────────────────────────────────────
  const stepMessages = [
    "DETECTING TEXT REGIONS…",
    "READING CHARACTERS…",
    "EXTRACTING ENTITIES…",
  ];
  const stepItems = document.querySelectorAll(".step-item");
  let stepTimer = null;

  function startSteps() {
    let s = 0;
    stepItems.forEach(el => { el.classList.remove("active", "done"); });
    loaderText.textContent = stepMessages[0];
    stepItems[0].classList.add("active");

    stepTimer = setInterval(() => {
      stepItems[s].classList.remove("active");
      stepItems[s].classList.add("done");
      s++;
      if (s < stepItems.length) {
        stepItems[s].classList.add("active");
        loaderText.textContent = stepMessages[s];
      }
    }, 2500);
  }

  function stopSteps() {
    clearInterval(stepTimer);
    stepItems.forEach(el => { el.classList.remove("active"); el.classList.add("done"); });
  }

  // ── Run pipeline ───────────────────────────────────────
  runBtn.addEventListener("click", runPipeline);

  async function runPipeline() {
    if (!currentFile) return;

    showState("loading");
    startSteps();
    runBtn.disabled = true;

    const fd = new FormData();
    fd.append("image", currentFile);

    try {
      const resp = await fetch("/predict", { method: "POST", body: fd });
      const data = await resp.json();
      stopSteps();

      if (!resp.ok || data.error) {
        showError(data.error || "Server error");
        return;
      }

      displayResults(data.fields, data.debug);
    } catch (err) {
      stopSteps();
      showError("Network error: " + err.message);
    }
  }

  function showError(msg) {
    runBtn.disabled = false;
    errorMsg.textContent = msg;
    showState("error");
  }

  retryBtn.addEventListener("click", () => {
    showState("idle");
    runBtn.disabled = false;
  });

  // ── Display results ────────────────────────────────────
  function displayResults(fields, debug) {
    lastFields = fields;

    const setField = (el, val) => {
      el.textContent = val || "—";
      const card = el.closest(".field-card");
      if (card) {
        if (val) {
          setTimeout(() => card.classList.add("populated"), 50);
        } else {
          card.classList.remove("populated");
        }
      }
    };

    setField(fCompany, fields.company);
    setField(fDate,    fields.date);
    setField(fAddress, fields.address);
    setField(fTotal,   fields.total ? `${fields.total}` : "");

    dBoxes.textContent = debug.num_boxes   ?? "—";
    dWords.textContent = debug.num_words   ?? "—";
    dOcr.textContent   = debug.ocr_preview ? debug.ocr_preview.slice(0, 200) + "…" : "—";

    showState("result");
    runBtn.disabled = false;
  }

  // ── Copy / Download ────────────────────────────────────
  copyBtn.addEventListener("click", () => {
    if (!lastFields) return;
    navigator.clipboard.writeText(JSON.stringify(lastFields, null, 2))
      .then(() => showToast("✓ Copied to clipboard"))
      .catch(() => showToast("✗ Copy failed"));
  });

  downloadBtn.addEventListener("click", () => {
    if (!lastFields) return;
    const blob = new Blob([JSON.stringify(lastFields, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "receipt_fields.json";
    a.click();
    URL.revokeObjectURL(a.href);
    showToast("✓ Downloading JSON");
  });

  // ── Init ───────────────────────────────────────────────
  showState("idle");
})();
