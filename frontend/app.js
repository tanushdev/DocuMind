/**
 * DocuMind Frontend Application
 * Handles document upload, chat interface, and API communication
 */

const API_BASE = 'http://localhost:8000';

// State
let documents = [];
let isProcessing = false;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const chatMessages = document.getElementById('chatMessages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const statusIndicator = document.getElementById('status');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkHealth();
    loadMetrics();

    // Periodic updates
    setInterval(loadMetrics, 10000);
});

function initializeEventListeners() {
    // File input
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Query input
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });
}

// Health Check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();

        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');

        if (data.status === 'ok') {
            statusDot.classList.add('connected');
            statusDot.classList.remove('error');
            statusText.textContent = 'Connected';
            enableChat();
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Degraded';
        }
    } catch (error) {
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');
        statusDot.classList.add('error');
        statusDot.classList.remove('connected');
        statusText.textContent = 'Disconnected';
    }
}

// File Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

async function handleFile(file) {
    // Validate file type
    const validTypes = ['.pdf', '.txt'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!validTypes.includes(ext)) {
        showError('Please upload a PDF or TXT file.');
        return;
    }

    // Validate file size (50MB max)
    if (file.size > 50 * 1024 * 1024) {
        showError('File too large. Maximum size is 50MB.');
        return;
    }

    await uploadFile(file);
}

async function uploadFile(file) {
    uploadArea.style.display = 'none';
    uploadProgress.style.display = 'block';
    progressFill.style.width = '10%';
    progressText.textContent = 'Uploading...';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/api/documents/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const data = await response.json();
        progressFill.style.width = '30%';
        progressText.textContent = 'Processing document...';

        // Poll for status
        await pollTaskStatus(data.task_id, file.name);

    } catch (error) {
        showError('Upload failed: ' + error.message);
        resetUpload();
    }
}

async function pollTaskStatus(taskId, filename) {
    const maxAttempts = 60;
    let attempts = 0;

    while (attempts < maxAttempts) {
        try {
            const response = await fetch(`${API_BASE}/api/documents/status/${taskId}`);
            const data = await response.json();

            // Update progress
            const progress = Math.min(30 + (data.progress || 0) * 70, 100);
            progressFill.style.width = `${progress}%`;
            progressText.textContent = formatStatus(data.status);

            if (data.status === 'completed') {
                // Success!
                addDocument({
                    id: data.document_id,
                    name: filename,
                    chunks: data.num_chunks
                });

                progressFill.style.width = '100%';
                progressText.textContent = 'Document processed successfully!';

                setTimeout(() => {
                    resetUpload();
                    addSystemMessage(`📄 Document "${filename}" has been processed (${data.num_chunks} chunks). You can now ask questions about it!`);
                    enableChat();
                }, 1500);

                return;
            } else if (data.status === 'failed') {
                throw new Error(data.error || 'Processing failed');
            }

            attempts++;
            await sleep(1000);

        } catch (error) {
            showError('Processing failed: ' + error.message);
            resetUpload();
            return;
        }
    }

    showError('Processing timed out');
    resetUpload();
}

function formatStatus(status) {
    const statusMap = {
        'extracting': 'Extracting text...',
        'chunking': 'Chunking document...',
        'embedding': 'Generating embeddings...',
        'indexing': 'Building search index...',
        'completed': 'Complete!',
        'failed': 'Failed'
    };
    return statusMap[status] || status;
}

function resetUpload() {
    uploadArea.style.display = 'block';
    uploadProgress.style.display = 'none';
    progressFill.style.width = '0%';
    fileInput.value = '';
}

// Chat
function enableChat() {
    queryInput.disabled = false;
    sendBtn.disabled = false;
    queryInput.placeholder = 'Ask a question about your documents...';
}

async function sendQuery() {
    const query = queryInput.value.trim();
    if (!query || isProcessing) return;

    isProcessing = true;
    queryInput.value = '';

    // Add user message
    addMessage(query, 'user');

    // Add loading message
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                top_k: 5
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Query failed');
        }

        const data = await response.json();

        // Remove loading message
        removeMessage(loadingId);

        // Add response
        addMessage(data.answer, 'assistant', data.sources, data.latency);

    } catch (error) {
        removeMessage(loadingId);
        addMessage(`Error: ${error.message}`, 'assistant');
    }

    isProcessing = false;
}

function addMessage(content, type, sources = null, latency = null) {
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.id = `msg-${Date.now()}`;

    let html = `<div class="message-content">${escapeHtml(content)}</div>`;

    if (sources && sources.length > 0) {
        html += `<div class="message-sources">
            <strong>Sources:</strong>
            ${sources.map((s, i) => `
                <span class="source-chip" title="${escapeHtml(s.chunk_text.substring(0, 200))}...">
                    [${i + 1}] ${s.document_id.substring(0, 8)}... (${(s.relevance_score * 100).toFixed(0)}%)
                </span>
            `).join('')}
        </div>`;
    }

    if (latency) {
        html += `<div class="message-latency">
            ⚡ ${latency.total_ms.toFixed(0)}ms 
            ${latency.cached ? '(cached)' : ''}
        </div>`;
    }

    messageDiv.innerHTML = html;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return messageDiv.id;
}

function addLoadingMessage() {
    const id = `loading-${Date.now()}`;
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = id;
    messageDiv.innerHTML = `
        <div class="loading-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeMessage(id) {
    const msg = document.getElementById(id);
    if (msg) msg.remove();
}

function addSystemMessage(content) {
    addMessage(content, 'assistant');
}

// Documents
function addDocument(doc) {
    documents.push(doc);
    updateDocumentList();
}

function updateDocumentList() {
    const list = document.getElementById('documentList');

    if (documents.length === 0) {
        list.innerHTML = '<p class="no-docs">No documents uploaded yet</p>';
        return;
    }

    list.innerHTML = documents.map(doc => `
        <div class="document-item">
            <span class="doc-icon">📄</span>
            <span class="doc-name" title="${doc.name}">${doc.name}</span>
            <span class="doc-chunks">${doc.chunks} chunks</span>
        </div>
    `).join('');
}

// Metrics
async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE}/metrics`);
        const data = await response.json();

        document.getElementById('vectorCount').textContent = data.vector_count.toLocaleString();
        document.getElementById('cacheHit').textContent = `${(data.cache_hit_ratio * 100).toFixed(0)}%`;
        document.getElementById('docCount').textContent = documents.length;

        // Average latency from total_ms if available
        if (data.stages && Object.keys(data.stages).length > 0) {
            const avgLatency = Object.values(data.stages)
                .filter(s => s.p50_ms)
                .reduce((sum, s) => sum + s.p50_ms, 0);
            document.getElementById('avgLatency').textContent = `${avgLatency.toFixed(0)}ms`;
        }
    } catch (error) {
        // Silently fail
    }
}

// Utilities
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showError(message) {
    alert(message); // Simple for now, could use a toast
}
