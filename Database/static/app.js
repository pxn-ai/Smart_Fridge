/**
 * Fridge Inventory — Frontend Logic
 * Handles fetching items, rendering the grid, CRUD modals, and filtering.
 */

// ── State ─────────────────────────────────────────
let allItems = [];
let currentFilter = "all";
let editingId = null;
let deleteId = null;
let simulateMode = false;
let cameraAvailable = false;
let captureLogLines = [];

// ── DOM refs ──────────────────────────────────────
const grid = document.getElementById("items-grid");
const emptyState = document.getElementById("empty-state");
const modalOverlay = document.getElementById("modal-overlay");
const deleteOverlay = document.getElementById("delete-overlay");
const form = document.getElementById("item-form");
const toast = document.getElementById("toast");

// ── Init ──────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadItems();
    bindEvents();
    checkCaptureStatus();
    startLivePreview();
    appendLog("Dashboard ready");
});

function bindEvents() {
    document.getElementById("btn-refresh").addEventListener("click", loadItems);
    document.getElementById("btn-add").addEventListener("click", openAddModal);
    document.getElementById("btn-capture").addEventListener("click", doCapture);
    document.getElementById("btn-modal-close").addEventListener("click", closeModal);
    document.getElementById("btn-cancel").addEventListener("click", closeModal);
    document.getElementById("btn-delete-cancel").addEventListener("click", closeDeleteModal);
    document.getElementById("btn-delete-confirm").addEventListener("click", confirmDelete);
    const clearLogButton = document.getElementById("btn-clear-log");
    if (clearLogButton) {
        clearLogButton.addEventListener("click", clearLog);
    }
    form.addEventListener("submit", handleSubmit);

    // Close modals on overlay click
    modalOverlay.addEventListener("click", (e) => {
        if (e.target === modalOverlay) closeModal();
    });
    deleteOverlay.addEventListener("click", (e) => {
        if (e.target === deleteOverlay) closeDeleteModal();
    });
    // Filter tabs
    document.querySelectorAll(".filter-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            document.querySelector(".filter-tab.active").classList.remove("active");
            tab.classList.add("active");
            currentFilter = tab.dataset.filter;
            renderItems();
        });
    });
}

async function checkCaptureStatus() {
    try {
        const res = await fetch('/api/capture/status');
        if (!res.ok) return;
        const data = await res.json();
        cameraAvailable = data.camera_available;
        const statusEl = document.getElementById('camera-status');
        const btn = document.getElementById('btn-simulate');
        const previewStatus = document.getElementById('preview-status');

        if (statusEl) statusEl.textContent = cameraAvailable ? 'Camera: Ready' : 'Camera: Unavailable';
        if (previewStatus) previewStatus.textContent = cameraAvailable ? 'Live feed' : 'Fallback preview';
        appendLog(cameraAvailable ? 'Camera status: ready' : 'Camera status: unavailable', cameraAvailable ? 'info' : 'warn');

        // Default simulate on if camera unavailable
        simulateMode = !cameraAvailable;
        if (btn) {
            btn.textContent = simulateMode ? 'Simulate: On' : 'Simulate: Off';
            btn.onclick = () => {
                simulateMode = !simulateMode;
                btn.textContent = simulateMode ? 'Simulate: On' : 'Simulate: Off';
            };
        }
    } catch (e) {
        console.warn('Could not get capture status', e);
    }
}

function appendLog(message, level = "info") {
    const logEl = document.getElementById("capture-log");
    if (!logEl) return;

    const timestamp = new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
    const prefix = level === "error" ? "[ERROR]" : level === "warn" ? "[WARN]" : "[INFO]";
    captureLogLines.push(`${timestamp} ${prefix} ${message}`);
    if (captureLogLines.length > 200) {
        captureLogLines = captureLogLines.slice(-200);
    }
    logEl.textContent = captureLogLines.join("\n");
    logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
    captureLogLines = [];
    appendLog("Log cleared");
}

function startLivePreview() {
    const previewImg = document.getElementById("live-preview-image");
    const previewStatus = document.getElementById("preview-status");
    if (!previewImg) return;

    previewImg.addEventListener("load", () => {
        if (previewStatus) {
            previewStatus.textContent = cameraAvailable ? "Live feed" : "Fallback preview";
        }
    });

    previewImg.addEventListener("error", () => {
        if (previewStatus) {
            previewStatus.textContent = "Preview error";
        }
    });

    previewImg.src = "/api/capture/stream";
    appendLog("Live preview started");
}

window.addEventListener("beforeunload", () => {
});

// ── API calls ─────────────────────────────────────
async function loadItems() {
    try {
        const [itemsRes, statsRes] = await Promise.all([
            fetch("/api/items"),
            fetch("/api/stats"),
        ]);
        const itemsData = await itemsRes.json();
        const statsData = await statsRes.json();

        allItems = itemsData.items;
        updateStats(statsData);
        renderItems();
    } catch (err) {
        showToast("Failed to load items");
        console.error(err);
    }
}

function updateStats(stats) {
    document.getElementById("val-total").textContent = stats.total;
    document.getElementById("val-fresh").textContent = stats.fresh;
    document.getElementById("val-warning").textContent = stats.expiring_soon;
    document.getElementById("val-expired").textContent = stats.expired;
}

// ── Rendering ─────────────────────────────────────
function renderItems() {
    const filtered =
        currentFilter === "all"
            ? allItems
            : allItems.filter((i) => i.expiry_status === currentFilter);

    if (filtered.length === 0) {
        grid.style.display = "none";
        emptyState.style.display = "flex";
    } else {
        grid.style.display = "grid";
        emptyState.style.display = "none";
    }

    grid.innerHTML = filtered.map((item, idx) => createCard(item, idx)).join("");
}

function createCard(item, idx) {
    const statusClass = {
        fresh: "expiry-fresh",
        warning: "expiry-warning",
        expired: "expiry-expired",
    }[item.expiry_status] || "expiry-fresh";

    const statusDot = {
        fresh: "🟢",
        warning: "🟠",
        expired: "🔴",
    }[item.expiry_status] || "⚪";

    let daysText = "";
    if (item.days_left !== null && item.days_left !== undefined) {
        if (item.days_left < 0) daysText = `${Math.abs(item.days_left)}d overdue`;
        else if (item.days_left === 0) daysText = "Expires today";
        else if (item.days_left === 1) daysText = "1 day left";
        else daysText = `${item.days_left} days left`;
    } else {
        daysText = item.expiry_date;
    }

    const displayName = item.name || `Item #${item.id}`;
    const formattedDate = formatDate(item.expiry_date);

    return `
    <div class="item-card" style="animation-delay:${idx * 0.04}s" data-status="${item.expiry_status}">
        <div class="item-img-wrap">
            <img src="/uploads/${encodeURIComponent(item.image_file)}"
                 alt="${escapeHtml(displayName)}"
                 loading="lazy"
                 onerror="this.parentElement.innerHTML='<div class=\\'item-img-placeholder\\'>📦</div>'">
        </div>
        <div class="item-body">
            <div class="item-name" title="${escapeHtml(displayName)}">${escapeHtml(displayName)}</div>
            <div class="item-id">ID: ${item.id} &middot; ${formattedDate}</div>
            <span class="item-expiry ${statusClass}">${statusDot} ${daysText}</span>
        </div>
        <div class="item-actions">
            <button class="btn btn-ghost btn-sm btn-icon" title="Edit" onclick="openEditModal(${item.id})">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                </svg>
            </button>
            <button class="btn btn-ghost btn-sm btn-icon" title="Delete" onclick="openDeleteModal(${item.id}, '${escapeHtml(displayName).replace(/'/g, "\\'")}')">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
            </button>
        </div>
    </div>`;
}

// ── Add / Edit Modal ──────────────────────────────
function openAddModal() {
    editingId = null;
    document.getElementById("modal-title").textContent = "Add Item";
    document.getElementById("btn-submit").textContent = "Add Item";
    form.reset();
    modalOverlay.classList.add("open");
}
// Expose globally for inline onclick
window.openAddModal = openAddModal;

function openEditModal(id) {
    const item = allItems.find((i) => i.id === id);
    if (!item) return;
    editingId = id;
    document.getElementById("modal-title").textContent = "Edit Item";
    document.getElementById("btn-submit").textContent = "Save Changes";
    document.getElementById("form-id").value = id;
    document.getElementById("form-name").value = item.name || "";
    document.getElementById("form-image").value = item.image_file;
    document.getElementById("form-expiry").value = item.expiry_date;
    modalOverlay.classList.add("open");
}
window.openEditModal = openEditModal;

function closeModal() {
    modalOverlay.classList.remove("open");
    editingId = null;
}

async function handleSubmit(e) {
    e.preventDefault();
    const data = {
        name: document.getElementById("form-name").value.trim(),
        image_file: document.getElementById("form-image").value.trim(),
        expiry_date: document.getElementById("form-expiry").value,
    };

    try {
        let res;
        if (editingId) {
            res = await fetch(`/api/items/${editingId}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        } else {
            res = await fetch("/api/items", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });
        }
        const result = await res.json();
        if (!res.ok) throw new Error(result.error);

        showToast(editingId ? "Item updated" : "Item added");
        closeModal();
        loadItems();
    } catch (err) {
        showToast(err.message || "Operation failed");
    }
}

// ── Delete Modal ──────────────────────────────────
function openDeleteModal(id, name) {
    deleteId = id;
    document.getElementById("delete-item-name").textContent = name;
    deleteOverlay.classList.add("open");
}
window.openDeleteModal = openDeleteModal;

function closeDeleteModal() {
    deleteOverlay.classList.remove("open");
    deleteId = null;
}

async function confirmDelete() {
    if (!deleteId) return;
    try {
        const res = await fetch(`/api/items/${deleteId}`, { method: "DELETE" });
        const result = await res.json();
        if (!res.ok) throw new Error(result.error);
        showToast("Item deleted");
        closeDeleteModal();
        loadItems();
    } catch (err) {
        showToast(err.message || "Delete failed");
    }
}

async function doCapture() {
    const itemName = "";
    
    try {
        appendLog("Camera capturing...");
        
        const res = await fetch("/api/capture", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: itemName, simulate: simulateMode }),
        });

        if (!res.ok) {
            const err = await res.json();
            appendLog(err.error || "Capture failed", "error");
            throw new Error(err.error || "Capture failed");
        }

        const result = await res.json();
        appendLog(`Captured image: ${result.image_file}`);

        // Show OCR text preview
        const ocrPreview = result.ocr_text
            ? result.ocr_text.replace(/\s+/g, ' ').trim().slice(0, 200)
            : 'No text detected';
        appendLog(`OCR text: ${ocrPreview}`);

        // Show confidence
        const confPct = (result.confidence * 100).toFixed(0);

        if (result.expiry_detected) {
            appendLog(`Detected expiry date: ${result.expiry_date} (${confPct}% confidence)`);
        } else {
            appendLog(`Detected expiry date: fallback ${result.expiry_date}`, "warn");
        }

        appendLog(`Saved item #${result.item_id} with confidence ${confPct}%`);

        showToast("✓ Item captured and added!");
        loadItems();

    } catch (err) {
        appendLog(`Capture failed: ${err.message}`, "error");
        showToast("✗ Capture failed: " + err.message);
    }
}

// ── Toast ─────────────────────────────────────────
function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2800);
}

// ── Helpers ───────────────────────────────────────
function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function formatDate(dateStr) {
    try {
        const d = new Date(dateStr + "T00:00:00");
        return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    } catch {
        return dateStr;
    }
}
