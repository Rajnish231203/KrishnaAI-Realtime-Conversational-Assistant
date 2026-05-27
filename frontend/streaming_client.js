/**
 * Real-Time Voice Assistant Client
 * Optimized for <1 second latency
 *
 * Features:
 * - 20-40ms audio chunking
 * - WebSocket streaming
 * - Real-time audio playback
 * - Barge-in support
 * - [NEW] Frontend multi-conversation session management
 */

// ── localStorage persistence constants ──────────────────────────
// Conversations are stored as passive finalized data only.
// Runtime state (websocket, playback, reveal, streaming) is NEVER persisted.
const STORAGE_KEY = 'krishna_conversations_v1';
const STORAGE_RETENTION_DAYS = 10;
// ────────────────────────────────────────────────────────────────

class StreamingVoiceClient {
    constructor() {
        this.ws = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.audioWorkletNode = null;
        this.isRecording = false;
        this.isSpeaking = false;
        this.totalChunksSent = 0;
        this.recordingStarted = false;
        this.holdTimerId = null;
        this.holdThresholdMs = 180;

        this.audioQueue = [];
        this.isPlaying = false;
        this.nextPlaybackTime = 0;
        this.leftoverBytes = null;

        // Metrics
        this.metrics = {
            sttStart: null,
            transcriptReady: null,
            llmStart: null,
            firstToken: null,
            ttsStart: null,
            playbackStart: null,
            responseComplete: null,
            tokenCount: 0,
            lastTokenAt: null
        };
        this._telemetryCache = {};

        // ── [NEW] Multi-conversation state ──────────────────────────
        // All conversations for this browser session (in-memory only).
        this.conversations = {};
        this.activeConversationId = null;
        // Tracks which conversation owns the current assistant stream.
        // Prevents token bleed when the user switches chats mid-stream.
        this.streamingConversationId = null;
        // ────────────────────────────────────────────────────────────

        // UI elements (IDs unchanged — required by existing JS)
        this.appEl = document.querySelector('.app');
        this.connectionText = document.getElementById('connectionText');
        this.connectionDot = document.getElementById('connectionDot');
        this.modeBadge = document.getElementById('modeBadge');
        this.talkBtn = document.getElementById('talkBtn');
        this.sendBtn = document.getElementById('sendBtn');
        this.textInput = document.getElementById('textInput');
        this.transcriptEl = document.getElementById('transcript');
        this.emptyStateEl = document.getElementById('emptyState');
        this.visualizerCanvas = document.getElementById('visualizer');
        this.visualizerCtx = this.visualizerCanvas.getContext('2d');

        // [NEW] Sidebar/topbar elements
        this.conversationListEl = document.getElementById('conversationList');
        this.newChatBtnEl = document.getElementById('newChatBtn');
        this.topbarChatTitleEl = document.getElementById('topbarChatTitle');
        this.clearChatsBtnEl = document.getElementById('clearChatsBtn');

        // Current transcript streaming refs
        this.currentUserTranscript = '';
        this.currentAssistantResponse = '';
        this.currentUserMessageEl = null;
        this.currentAssistantMessageEl = null;

        // ── Synchronized text reveal state ──────────────────────────
        // Progressive word-by-word reveal synced with audio playback.
        this._revealTimerId = null;          // setInterval handle
        this._revealWords = [];              // full word array to reveal
        this._revealIndex = 0;               // next word index to show
        this._revealFullText = '';           // complete response text
        this._totalAudioDuration = 0;       // REAL accumulated audio duration (seconds)
        this._responseComplete = false;      // true after response_complete fires
        this._isVoiceTurn = false;           // true for voice turns, false for text chat
        this._scrollRafPending = false;      // rAF scroll throttle guard
        this._audioPlaybackStarted = false;  // true once first audio chunk arrives
        this._audioPlaybackPrepared = false; // true once first buffer is scheduled in playAudioQueue
        this._awaitingPlaybackCleanup = false; // true when voice response awaits drain cleanup
        // ─────────────────────────────────────────────────────────────

        // ── WebSocket reconnect state ──────────────────────────────
        // Handles startup race, backend restart, and network interruption.
        // Reconnect is transport-only — never touches conversations,
        // playback, reveal, persistence, or telemetry state.
        this._reconnectAttempts = 0;
        this._reconnectTimerId = null;       // single-timer guard
        this._reconnectDelayMs = 3000;       // base delay (exponential backoff)
        this._isManualDisconnect = false;    // true only on intentional close
        // ─────────────────────────────────────────────────────────────

        this.init();
    }

    async init() {
        this.setAppState('idle');

        // Initialize audio context
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });

        // Gain + compressor to prevent distortion
        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.value = 0.95;

        this.compressor = this.audioContext.createDynamicsCompressor();
        this.compressor.threshold.setValueAtTime(-20, this.audioContext.currentTime);
        this.compressor.knee.setValueAtTime(40, this.audioContext.currentTime);
        this.compressor.ratio.setValueAtTime(12, this.audioContext.currentTime);
        this.compressor.attack.setValueAtTime(0, this.audioContext.currentTime);
        this.compressor.release.setValueAtTime(0.25, this.audioContext.currentTime);
        this.compressor.connect(this.gainNode);
        this.gainNode.connect(this.audioContext.destination);

        // Connect WebSocket
        this.connectWebSocket();

        // Setup UI interactions
        this.setupUI();

        // Setup visualizer
        this.setupVisualizer();

        // [NEW] Try restoring persisted conversations before creating a fresh one.
        // Persistence is PASSIVE STORAGE ONLY — only finalized messages are restored.
        // Runtime state (websocket, playback, streaming, reveal) is never persisted.
        const restored = this.loadPersistedConversations();
        if (!restored) {
            this.createNewConversation();
        }

        // Telemetry defaults
        this._initTelemetryPanel();
    }

    connectWebSocket() {
        // Guard: if a previous socket is still OPEN or CONNECTING, skip.
        if (this.ws && (this.ws.readyState === WebSocket.OPEN ||
                        this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        const isReconnect = this._reconnectAttempts > 0;
        const statusLabel = isReconnect ? 'Reconnecting' : 'Connecting';
        this.updateConnectionStatus(statusLabel, false);
        this.updateStatus(`${statusLabel}...`, 'idle');

        const wsUrl = this._resolveWebSocketUrl();

        try {
            this.ws = new WebSocket(wsUrl);
        } catch (err) {
            // Constructor can throw on invalid URL — schedule retry.
            console.warn('❌ WebSocket constructor failed:', err.message);
            this._scheduleReconnect();
            return;
        }

        // ── onopen ─────────────────────────────────────────────────
        this.ws.onopen = () => {
            console.log('✅ Connected to server' +
                (isReconnect ? ` (after ${this._reconnectAttempts} retries)` : ''));

            // Clear reconnect state
            this._reconnectAttempts = 0;
            if (this._reconnectTimerId !== null) {
                clearTimeout(this._reconnectTimerId);
                this._reconnectTimerId = null;
            }

            this.updateConnectionStatus('Connected', true);
            this.updateStatus('Ready', 'idle');
            this.talkBtn.disabled = false;
            this.updateMicText('idle');

            // Re-sync active conversation with backend after reconnect.
            // This is a lightweight fire-and-forget signal so the backend
            // knows which conversation context to use for voice turns.
            if (isReconnect && this.activeConversationId) {
                try {
                    this.ws.send(JSON.stringify({
                        type: 'switch_conversation',
                        conversation_id: this.activeConversationId,
                    }));
                } catch (e) { /* non-critical */ }
            }
        };

        // ── onmessage ──────────────────────────────────────────────
        this.ws.onmessage = (event) => {
            this.handleServerMessage(JSON.parse(event.data));
        };

        // ── onerror ────────────────────────────────────────────────
        // Errors are logged only. Reconnect logic lives exclusively
        // in onclose to prevent duplicate reconnect loops.
        this.ws.onerror = (error) => {
            console.warn('⚠️ WebSocket error:', error);
        };

        // ── onclose ───────────────────────────────────────────────
        // Handles both unexpected disconnects (backend restart, network
        // interruption) and startup race (backend not ready yet).
        // Frontend state (conversations, playback, persistence) is
        // NEVER touched here — only transport recovery.
        this.ws.onclose = (event) => {
            console.log(`🔌 WebSocket closed (code=${event.code}, reason=${event.reason || 'none'})`);

            this.talkBtn.disabled = true;
            this.updateMicText('disconnected');

            if (this._isManualDisconnect) {
                // Intentional disconnect — don't reconnect.
                this.updateConnectionStatus('Disconnected', false);
                this.updateStatus('Disconnected', 'error');
                return;
            }

            // Unexpected disconnect — schedule automatic reconnect.
            this.updateConnectionStatus('Reconnecting', false);
            this.updateStatus('Reconnecting...', 'idle');
            this._scheduleReconnect();
        };
    }

    _resolveWebSocketUrl() {
        if (window.WS_URL) return window.WS_URL;

        const isLocal = this._isLocalhostHost();
        const wsHost = window.WS_HOST || window.location.hostname;

        if (isLocal) {
            const wsPort = window.WS_PORT || 8766;
            return `ws://${wsHost}:${wsPort}`;
        }

        if (window.WS_HOST || window.WS_PORT) {
            const wsPort = window.WS_PORT ? `:${window.WS_PORT}` : '';
            return `wss://${wsHost}${wsPort}`;
        }

        const prodUrl = window.WS_PROD_URL || 'wss://krishnaai-realtime-conversational-assistant-production.up.railway.app';
        return prodUrl;
    }

    _isLocalhostHost() {
        const host = window.location.hostname;
        return host === 'localhost' || host === '127.0.0.1';
    }

    // ══════════════════════════════════════════════════════════════
    // WEBSOCKET RECONNECT — TRANSPORT RECOVERY ONLY
    // Never touches conversations, playback, reveal, or persistence.
    // ══════════════════════════════════════════════════════════════

    /**
     * _scheduleReconnect()
     * Schedules a single delayed reconnect attempt.
     * Uses a timer guard (_reconnectTimerId) to prevent reconnect storms —
     * only ONE timer can exist at a time.
     */
    _scheduleReconnect() {
        // Single-timer guard: prevent duplicate reconnect timers.
        if (this._reconnectTimerId !== null) return;

        this._reconnectAttempts++;
        const delay = this._getReconnectDelay();

        console.log(
            `🔄 Reconnect attempt #${this._reconnectAttempts} in ${(delay / 1000).toFixed(1)}s`
        );

        this._reconnectTimerId = setTimeout(() => {
            this._reconnectTimerId = null;
            this.connectWebSocket();
        }, delay);
    }

    /**
     * _getReconnectDelay()
     * Exponential backoff: 3s → 6s → 12s → 24s → 30s (cap).
     * First attempt uses base delay for fast recovery.
     */
    _getReconnectDelay() {
        const base = this._reconnectDelayMs;              // 3000ms
        const maxDelay = 30000;                           // 30s cap
        const delay = base * Math.pow(2, Math.min(this._reconnectAttempts - 1, 4));
        return Math.min(delay, maxDelay);
    }

    setupUI() {
        // Hold-to-talk (Pointer Events)
        this.talkBtn.addEventListener('pointerdown', (e) => this.handlePointerDown(e));
        this.talkBtn.addEventListener('pointerup', (e) => this.handlePointerUp(e));
        this.talkBtn.addEventListener('pointerleave', (e) => this.handlePointerUp(e));
        this.talkBtn.addEventListener('pointercancel', (e) => this.handlePointerUp(e));

        // Text send
        this.sendBtn.addEventListener('click', () => this.sendChatMessage());
        this.textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendChatMessage();
            }
        });

        // [NEW] New Chat button
        if (this.newChatBtnEl) {
            this.newChatBtnEl.addEventListener('click', () => {
                this.createNewConversation();
                // Close mobile sidebar after creating
                document.getElementById('sidebar')?.classList.remove('open');
                document.getElementById('sidebarOverlay')?.classList.remove('visible');
            });
        }

        // [NEW] Clear Conversations button
        if (this.clearChatsBtnEl) {
            this.clearChatsBtnEl.addEventListener('click', () => {
                this.clearAllConversations();
                document.getElementById('sidebar')?.classList.remove('open');
                document.getElementById('sidebarOverlay')?.classList.remove('visible');
            });
        }
    }

    setupVisualizer() {
        this.visualizerCanvas.width = this.visualizerCanvas.offsetWidth;
        this.visualizerCanvas.height = this.visualizerCanvas.offsetHeight;

        window.addEventListener('resize', () => {
            this.visualizerCanvas.width = this.visualizerCanvas.offsetWidth;
            this.visualizerCanvas.height = this.visualizerCanvas.offsetHeight;
        });

        this.animateVisualizer();
    }

    animateVisualizer() {
        const draw = () => {
            const ctx = this.visualizerCtx;
            const width = this.visualizerCanvas.width;
            const height = this.visualizerCanvas.height;

            ctx.clearRect(0, 0, width, height);

            if (this.isRecording || this.isSpeaking) {
                const time = Date.now() / 1000;
                const bars = 18;
                const barWidth = width / bars;
                const activeColor = this.isRecording
                    ? 'rgba(217, 119, 6, 0.8)'
                    : 'rgba(37, 99, 235, 0.75)';

                for (let i = 0; i < bars; i++) {
                    const barHeight = Math.sin(time * 3 + i * 0.5) * 8 + 12;
                    const x = i * barWidth;
                    const y = (height - barHeight) / 2;
                    ctx.fillStyle = activeColor;
                    ctx.fillRect(x, y, barWidth - 2, barHeight);
                }
            }

            requestAnimationFrame(draw);
        };

        draw();
    }

    // ══════════════════════════════════════════════════════════════
    // [NEW] MULTI-CONVERSATION MANAGEMENT
    // ══════════════════════════════════════════════════════════════

    /**
     * createNewConversation()
     * Creates a fresh in-memory conversation, sets it as active,
     * clears the transcript UI and resets streaming state.
     * Does NOT touch the WebSocket connection.
     */
    createNewConversation() {
        const id = 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7);

        this.conversations[id] = {
            id,
            title: 'New conversation',
            messages: []
        };

        // Switch to new conversation (handles UI reset + sidebar render)
        this.switchConversation(id);

        // Persist new conversation to localStorage.
        this.persistConversations();
    }

    /**
     * switchConversation(conversationId)
     * Stops any in-progress audio, resets streaming state,
     * and re-renders the transcript for the selected conversation.
     * Does NOT restart or reconnect the WebSocket.
     */
    switchConversation(conversationId) {
        if (this.isRecording) {
            this.updateStatus('Finish speaking before switching chat', 'idle');
            return;
        }
        if (!this.conversations[conversationId]) return;

        // Stop audio pipeline first — clears queue, stops current source.
        this.stopAudioPlayback();

        // Explicit speaking/playing state reset after stopAudioPlayback.
        this.isSpeaking = false;
        this.isPlaying = false;

        // Full stream ownership + DOM ref cleanup.
        this.currentAssistantResponse = '';
        this.currentAssistantMessageEl = null;
        this.currentUserMessageEl = null;
        this.currentUserTranscript = '';
        this.streamingConversationId = null;

        // Set active
        this.activeConversationId = conversationId;

        // Notify backend so voice turns use the correct history container.
        // This is a lightweight fire-and-forget signal — no ack required.
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'switch_conversation',
                conversation_id: conversationId,
            }));
        }

        // Update topbar title
        this._updateTopbarTitle();

        // Re-render sidebar active states
        this.renderConversationList();

        // Re-render transcript for this conversation
        this.renderActiveConversation();

        // Persist active conversation selection.
        this.persistConversations();
    }

    /**
     * renderConversationList()
     * Renders sidebar chat items. Marks active conversation.
     */
    renderConversationList() {
        if (!this.conversationListEl) return;

        this.conversationListEl.innerHTML = '';

        // Render in reverse-chronological order (newest first)
        const ids = Object.keys(this.conversations).reverse();

        if (ids.length === 0) {
            const placeholder = document.createElement('div');
            placeholder.style.cssText = 'padding:12px 12px;font-size:0.78rem;color:var(--text-muted);';
            placeholder.textContent = 'No conversations yet.';
            this.conversationListEl.appendChild(placeholder);
            return;
        }

        for (const id of ids) {
            const conv = this.conversations[id];
            const item = document.createElement('div');
            item.className = 'chat-item' + (id === this.activeConversationId ? ' active' : '');
            item.dataset.convId = id;

            const icon = document.createElement('span');
            icon.className = 'chat-icon';
            icon.textContent = '💬';

            const title = document.createElement('span');
            title.className = 'chat-title';
            title.textContent = conv.title;

            item.appendChild(icon);
            item.appendChild(title);

            item.addEventListener('click', () => {
                this.switchConversation(id);
                // Close mobile sidebar
                document.getElementById('sidebar')?.classList.remove('open');
                document.getElementById('sidebarOverlay')?.classList.remove('visible');
            });

            this.conversationListEl.appendChild(item);
        }
    }

    /**
     * renderActiveConversation()
     * Clears the transcript container and re-renders all finalized
     * messages for the current active conversation.
     *
     * RENDER SAFETY: DOM refs are nullified here — not just in
     * switchConversation — because this is the exact point where
     * existing nodes are removed. Any handler that fires between
     * switchConversation and renderActiveConversation completing
     * will get null refs instead of dangling pointers.
     */
    renderActiveConversation() {
        if (!this.transcriptEl) return;

        // Null out streaming DOM refs BEFORE removing nodes.
        // This prevents any in-flight handler from writing into
        // a detached element after the querySelectorAll sweep below.
        this.currentAssistantMessageEl = null;
        this.currentUserMessageEl = null;

        // Remove all rendered message elements (preserves #emptyState)
        const existingMessages = this.transcriptEl.querySelectorAll('.message');
        existingMessages.forEach(el => el.remove());

        const conv = this.conversations[this.activeConversationId];
        if (!conv) return;

        if (conv.messages.length === 0) {
            // Show empty state
            if (this.emptyStateEl) {
                this.emptyStateEl.classList.remove('hidden');
            }
            return;
        }

        // Hide empty state
        if (this.emptyStateEl) {
            this.emptyStateEl.classList.add('hidden');
        }

        // Render stored messages
        for (const msg of conv.messages) {
            const el = this.createMessageElement(msg.role, msg.content, false);
            this.transcriptEl.appendChild(el);
        }

        this.scrollToBottom();
    }

    /**
     * _updateTopbarTitle()
     * Syncs the topbar chat title with the active conversation.
     */
    _updateTopbarTitle() {
        if (!this.topbarChatTitleEl) return;
        const conv = this.conversations[this.activeConversationId];
        this.topbarChatTitleEl.textContent = conv ? conv.title : 'New conversation';
        this._updateActiveConversationTelemetry();
    }

    /**
     * _deriveTitle(text)
     * Derives a short display title from the first user message.
     */
    _deriveTitle(text) {
        if (!text) return 'New conversation';
        const clean = text.trim().replace(/\s+/g, ' ');
        return clean.length > 38 ? clean.slice(0, 36).trimEnd() + '…' : clean;
    }

    /**
     * _saveMessage(role, content)
     * Stores a finalized message in the active conversation object
     * and updates the sidebar title on first user message.
     */
    _saveMessage(role, content) {
        if (!this.activeConversationId) return;
        const conv = this.conversations[this.activeConversationId];
        if (!conv) return;

        conv.messages.push({ role, content });

        // Use first user message as title
        if (role === 'user' && conv.title === 'New conversation') {
            conv.title = this._deriveTitle(content);
            this._updateTopbarTitle();
            this.renderConversationList();
            this._updateActiveConversationTelemetry();
        }

        // Persist finalized message to localStorage.
        // This is the primary persistence hook — messages become
        // stable here and are safe to write to storage.
        this.persistConversations();
    }

    // ══════════════════════════════════════════════════════════════
    // LOCALSTORAGE PERSISTENCE — PASSIVE STORAGE ONLY
    // Stores finalized conversation history. Never stores runtime
    // state (websocket, playback, reveal, streaming, audio queue).
    // ══════════════════════════════════════════════════════════════

    /**
     * persistConversations()
     * Serializes finalized conversation state to localStorage.
     * Wrapped in try/catch for quota exceeded, private browsing,
     * and disabled storage scenarios — failures are logged only,
     * never crash the app.
     */
    persistConversations() {
        try {
            const payload = {
                version: 1,
                savedAt: Date.now(),
                activeConversationId: this.activeConversationId,
                conversations: this.conversations
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
        } catch (err) {
            console.warn('⚠️ Failed to persist conversations:', err.message);
        }
    }

    /**
     * loadPersistedConversations()
     * Safely restores persisted chats from localStorage.
     * Validates schema version, JSON integrity, and retention window.
     * Returns true if conversations were successfully restored.
     *
     * Restores ONLY: this.conversations, this.activeConversationId
     * Does NOT restore: websocket, playback, streaming, reveal state
     */
    loadPersistedConversations() {
        try {
            // CHECK 1 — data exists
            const raw = localStorage.getItem(STORAGE_KEY);
            if (!raw) return false;

            // CHECK 2 — valid JSON
            let data;
            try {
                data = JSON.parse(raw);
            } catch (parseErr) {
                console.warn('⚠️ Corrupted conversation storage — clearing.', parseErr.message);
                localStorage.removeItem(STORAGE_KEY);
                return false;
            }

            // CHECK 3 — version validation
            if (!data || data.version !== 1) {
                console.warn('⚠️ Incompatible storage schema (version mismatch) — clearing.');
                localStorage.removeItem(STORAGE_KEY);
                return false;
            }

            // CHECK 4 — retention window
            const ageMs = Date.now() - (data.savedAt || 0);
            const retentionMs = STORAGE_RETENTION_DAYS * 24 * 60 * 60 * 1000;
            if (ageMs > retentionMs) {
                console.info('ℹ️ Stored conversations expired (%d days old) — clearing.',
                    Math.floor(ageMs / (24 * 60 * 60 * 1000)));
                localStorage.removeItem(STORAGE_KEY);
                return false;
            }

            // CHECK 5 — conversations object exists and is non-empty
            if (!data.conversations || typeof data.conversations !== 'object' ||
                Object.keys(data.conversations).length === 0) {
                console.warn('⚠️ No valid conversations in storage — clearing.');
                localStorage.removeItem(STORAGE_KEY);
                return false;
            }

            // ── RESTORE FINALIZED STATE ONLY ──
            this.conversations = data.conversations;

            // Validate that activeConversationId still exists in restored data
            if (data.activeConversationId && this.conversations[data.activeConversationId]) {
                this.activeConversationId = data.activeConversationId;
            } else {
                // Fall back to most recent conversation
                const ids = Object.keys(this.conversations);
                this.activeConversationId = ids[ids.length - 1];
            }

            // Render restored state (no runtime state restored)
            this.renderConversationList();
            this.renderActiveConversation();
            this._updateTopbarTitle();

            console.info('✅ Restored %d conversation(s) from localStorage.',
                Object.keys(this.conversations).length);
            return true;

        } catch (err) {
            console.warn('⚠️ Failed to load persisted conversations:', err.message);
            return false;
        }
    }

    /**
     * clearAllConversations()
     * Removes all persisted data and resets to a fresh conversation.
     * Used by the "Clear Conversations" sidebar button.
     */
    clearAllConversations() {
        try {
            localStorage.removeItem(STORAGE_KEY);
        } catch (err) {
            console.warn('⚠️ Failed to clear localStorage:', err.message);
        }

        // Stop any in-progress audio/streaming
        this.stopAudioPlayback();

        // Reset in-memory state
        this.conversations = {};
        this.activeConversationId = null;
        this.streamingConversationId = null;
        this.currentAssistantResponse = '';
        this.currentAssistantMessageEl = null;
        this.currentUserMessageEl = null;
        this.currentUserTranscript = '';

        // Create fresh conversation
        this.createNewConversation();
    }

    // ══════════════════════════════════════════════════════════════
    // RECORDING (unchanged logic, no WS changes)
    // ══════════════════════════════════════════════════════════════

    async startRecording() {
        if (this.isRecording) return;

        console.log('🎤 Starting recording');

        if (this.isSpeaking) {
            this.ws.send(JSON.stringify({ type: 'interrupt' }));
            this.stopAudioPlayback();
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
            console.log('🔊 AudioContext resumed');
        }

        this.isRecording = true;
        this.talkBtn.classList.add('recording');

        this._resetTelemetryForNewResponse(false);
        this.updateTelemetryMetric('metricStreamingState', 'Voice');
        this.updateTelemetryMetric('metricPlayback', 'Idle');

        this.currentUserTranscript = '';
        this.totalChunksSent = 0;
        this._voiceUserSavedThisTurn = false; // FIX: reset duplicate-save guard per turn

        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            const bufferSize = 1024;
            const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

            let chunkBuffer = [];
            let chunkCount = 0;

            processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;

                const inputData = e.inputBuffer.getChannelData(0);
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }

                chunkBuffer.push(pcmData);
                chunkCount++;

                if (chunkCount >= 2) {
                    this.sendAudioChunk(chunkBuffer);
                    if (this.totalChunksSent % 25 === 0) {
                        console.log(`📤 Chunks sent: ${this.totalChunksSent}`);
                    }
                    chunkBuffer = [];
                    chunkCount = 0;
                }
            };

            source.connect(processor);
            processor.connect(this.audioContext.destination);

            this.audioProcessor = processor;
            this.audioSource = source;
            this.recordingStarted = true;
            this.setAppState('listening');
            this.updateStatus('Listening...', 'listening');
            this.updateMicText('listening');

        } catch (error) {
            console.error('❌ Microphone access error:', error);
            this.updateStatus('Microphone access denied', '');
            this.isRecording = false;
            this.talkBtn.classList.remove('recording');
        }
    }

    stopRecording() {
        if (!this.isRecording) return;

        console.log('🛑 Stopping recording');

        this.isRecording = false;
        this.talkBtn.classList.remove('recording');

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }

        if (this.audioProcessor) this.audioProcessor.disconnect();
        if (this.audioSource) this.audioSource.disconnect();

        this.recordingStarted = false;

        if (this.totalChunksSent === 0) {
            this.setAppState('idle'); // FIX: reset visualizer/state on tap-without-speech
            this.updateStatus('Ready', 'idle');
            this.updateMicText('idle');
            this.updateTelemetryMetric('metricStreamingState', 'Idle');
            return;
        }

        this.metrics.sttStart = performance.now();

        this.setAppState('processing');
        this.updateStatus('Processing...', 'processing');
        this.ws.send(JSON.stringify({ type: 'end_of_speech' }));
    }

    sendAudioChunk(chunks) {
        const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const combined = new Int16Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
            combined.set(chunk, offset);
            offset += chunk.length;
        }

        const bytes = new Uint8Array(combined.buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        const base64 = btoa(binary);

        this.ws.send(JSON.stringify({ type: 'audio_chunk', audio: base64 }));
        this.totalChunksSent++;
    }

    // ══════════════════════════════════════════════════════════════
    // SERVER MESSAGE HANDLER (unchanged protocol)
    // ══════════════════════════════════════════════════════════════

    handleServerMessage(data) {
        switch (data.type) {
            case 'transcript_partial':
                this.handlePartialTranscript(data.text);
                break;

            case 'transcript_final':
                this.handleFinalTranscript(data.text);
                break;

            case 'llm_token':
                this.handleLLMToken(data.token);
                break;

            case 'audio_response_chunk':
                this.handleAudioChunk(data.audio);
                break;

            case 'response_complete':
                this.handleResponseComplete();
                break;

            case 'state':
                // Frontend owns lifecycle UI states.
                // Backend state messages are now limited to:
                // - errors
                // - warnings
                // - operational notifications
                //
                // This prevents backend/frontend status desynchronization
                // where delayed backend events overwrite the real frontend state.
                if (data.status === 'error' || data.status === 'warning') {
                    this.updateStatus(data.message, data.status || 'idle');
                }
                break;
        }
    }

    // ══════════════════════════════════════════════════════════════
    // TRANSCRIPT HANDLERS
    // [NEW] _saveMessage() calls added to persist finalized messages
    // ══════════════════════════════════════════════════════════════

    handlePartialTranscript(text) {
        if (text.trim().length <= 1) return;
        this.currentUserTranscript = text;
        this.updateTranscript('user', text, true);
    }

    handleFinalTranscript(text) {
        // Upgrade the partial user bubble to its final form in the DOM.
        // updateTranscript() will find the existing partial element and
        // promote it — no new node is created, so no duplicate appears.
        this.currentUserTranscript = text;
        this.updateTranscript('user', text, false);

        this.metrics.transcriptReady = performance.now();
        this.metrics.llmStart = performance.now();
        this.updateMetrics();

        // VOICE PERSISTENCE SAFETY:
        // _voiceUserSavedThisTurn is reset to false at the start of every
        // recording turn (startRecording). It is set true here the moment we
        // persist, so if the server ever emits a duplicate transcript_final
        // for the same turn (edge case), the second one is silently ignored.
        // This flag is intentionally separate from DOM state — it protects
        // the conversation memory store, not the UI. When renderActiveConversation
        // rebuilds the view, it reads only from conversation.messages[], so
        // message correctness depends entirely on what we save here.
        if (!this._voiceUserSavedThisTurn) {
            this._saveMessage('user', text);
            this._voiceUserSavedThisTurn = true;
        }

        // Lock the assistant stream to this conversation.
        // Both token and audio handlers check this before processing,
        // so switching chats mid-response cannot corrupt either channel.
        this.streamingConversationId = this.activeConversationId;

        // Mark this as a voice turn — text reveal will be synchronized
        // with audio playback instead of rendered immediately.
        this._isVoiceTurn = true;
        this._responseComplete = false;
        this._totalAudioDuration = 0;
        this._audioPlaybackStarted = false;
        this._audioPlaybackPrepared = false;
        this._awaitingPlaybackCleanup = false;

        this.currentAssistantResponse = '';
        this.currentAssistantMessageEl = null;
        this.updateStatus('Krishna is thinking...', 'processing');
        this.setAppState('processing');
    }

    handleLLMToken(token) {
        // FIX: Ignore tokens that belong to a different (now-inactive) conversation.
        // This prevents assistant responses from bleeding into a switched chat.
        if (this.streamingConversationId !== this.activeConversationId) return;

        const now = performance.now();
        if (this.metrics.llmStart === null) {
            this.metrics.llmStart = now;
        }
        if (this.metrics.firstToken === null) {
            this.metrics.firstToken = now;
            this.updateMetrics();
        }
        this.metrics.tokenCount += 1;
        this.metrics.lastTokenAt = now;
        this._updateTokensPerSec(now);

        this.currentAssistantResponse += token;

        if (this._isVoiceTurn) {
            // VOICE MODE: Do NOT immediately render text.
            // Text will be progressively revealed in sync with audio playback.
            // Create an empty assistant bubble placeholder if it doesn't exist yet.
            if (!this.currentAssistantMessageEl) {
                this.updateTranscript('assistant', '', true);
            }
        } else {
            // TEXT MODE: Render tokens immediately for live streaming UX.
            this.updateTranscript('assistant', this.currentAssistantResponse, true);
        }
    }

    async handleAudioChunk(base64Audio) {
        // AUDIO OWNERSHIP GUARD — must be the very first check.
        // If the audio chunk belongs to a conversation the user has already
        // switched away from, drop it completely. This closes the TTS-bleed
        // gap that token ownership alone cannot cover: even with tokens
        // discarded, audio PCM chunks were still queued and played globally.
        if (this.streamingConversationId !== this.activeConversationId) return;

        if (!this._audioPlaybackStarted) {
            this.updateTelemetryMetric('metricPlayback', 'Buffering');
            this.updateStatus('Krishna is speaking...', 'speaking');
            this.isSpeaking = true;
            this.setAppState('speaking');

            // Mark that audio chunks have started arriving.
            // Reveal does NOT start here — it waits for _audioPlaybackPrepared
            // (set in playAudioQueue after first buffer is scheduled).
            this._audioPlaybackStarted = true;
        }

        const binaryString = atob(base64Audio);
        let bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        if (this.leftoverBytes) {
            const combined = new Uint8Array(this.leftoverBytes.length + bytes.length);
            combined.set(this.leftoverBytes);
            combined.set(bytes, this.leftoverBytes.length);
            bytes = combined;
            this.leftoverBytes = null;
        }

        if (bytes.length % 2 !== 0) {
            this.leftoverBytes = bytes.slice(-1);
            bytes = bytes.slice(0, -1);
        }

        if (bytes.length === 0) return;

        const pcmData = new Int16Array(bytes.buffer, 0, bytes.length / 2);
        this.audioQueue.push(pcmData);
        this.updateTelemetryMetric('metricAudioQueue', `${this.audioQueue.length}`);

        if (!this.isPlaying && this.audioQueue.length >= 1) {
            this.playAudioQueue();
        }
    }

    async playAudioQueue() {
        this.isPlaying = true;
        const sampleRate = 16000;

        while (this.audioQueue.length > 0) {
            const pcmData = this.audioQueue.shift();
            this.updateTelemetryMetric('metricAudioQueue', `${this.audioQueue.length}`);
            const floatData = new Float32Array(pcmData.length);
            for (let i = 0; i < pcmData.length; i++) {
                floatData[i] = pcmData[i] / 32768.0;
            }

            try {
                const buffer = this.audioContext.createBuffer(1, floatData.length, sampleRate);
                buffer.copyToChannel(floatData, 0);

                // Accumulate REAL audio duration for synchronized reveal timing.
                this._totalAudioDuration += buffer.duration;

                const source = this.audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(this.compressor);
                this.currentSource = source;

                const now = this.audioContext.currentTime;
                if (this.nextPlaybackTime < now) {
                    this.nextPlaybackTime = now + 0.05;
                }

                source.start(this.nextPlaybackTime);
                this.nextPlaybackTime += buffer.duration;

                if (!this._audioPlaybackPrepared) {
                    this._audioPlaybackPrepared = true;
                    this.metrics.playbackStart = performance.now();
                    this.metrics.ttsStart = this.metrics.playbackStart;
                    this.updateTelemetryMetric('metricPlayback', 'Playing');
                    this.updateMetrics();
                    this._tryStartSynchronizedReveal();
                }


                await new Promise(r => setTimeout(r, 0));
            } catch (error) {
                console.error('❌ PCM playback error:', error);
            }
        }

        this.isPlaying = false;
        if (!this.isRecording) {
            // Assistant lifecycle fully completed.
            // Audio playback queue drained successfully.
            this.isSpeaking = false;
            this.updateStatus('Ready', 'idle');
            this.setAppState('idle');
            this.updateMicText('idle');
            this.updateTelemetryMetric('metricPlayback', 'Idle');
            this.updateTelemetryMetric('metricStreamingState', 'Idle');

            if (this._awaitingPlaybackCleanup) {
                this.currentAssistantResponse = '';
                this.currentAssistantMessageEl = null;
                this.streamingConversationId = null;
                this._awaitingPlaybackCleanup = false;
            }

            const totalStart = this.metrics.sttStart ?? this.metrics.llmStart;
            if (totalStart !== null) {
                const totalLatency = performance.now() - totalStart;
                this.updateTelemetryMetric('metricTotalLatency', this._formatMs(totalLatency));
            }
        }
    }

    stopAudioPlayback() {
        this.audioQueue = [];
        this.isPlaying = false;
        this.isSpeaking = false;
        this.updateTelemetryMetric('metricAudioQueue', '0');
        this.updateTelemetryMetric('metricPlayback', 'Stopped');

        // Stop synchronized text reveal immediately on interruption.
        this._stopSynchronizedReveal();
        this._audioPlaybackStarted = false;
        this._audioPlaybackPrepared = false;
        this._awaitingPlaybackCleanup = false;

        // Forced interruption — release ownership immediately.
        // Unlike normal drain (playAudioQueue), interruption
        // means playback will never naturally complete.
        this.currentAssistantResponse = '';
        this.currentAssistantMessageEl = null;
        this.streamingConversationId = null;

        // Playback interruption cleanup.
        // Ensure lifecycle UI fully resets after
        // forced stop / barge-in / conversation switch.
        this.setAppState('idle');
        this.updateStatus('Ready', 'idle');
        this.updateMicText('idle');
        this.updateTelemetryMetric('metricStreamingState', 'Idle');

        this.nextPlaybackTime = 0;
        this.leftoverBytes = null;

        if (this.currentSource) {
            try { this.currentSource.stop(); } catch (e) { }
            this.currentSource = null;
        }
    }

    handleResponseComplete() {
        // ── SEMANTIC NOTE ──────────────────────────────────────────────
        // response_complete indicates LLM/TTS GENERATION completion,
        // NOT audio playback completion. Audio may still be buffered
        // and playing via the audioContext scheduler. Playback
        // completion is handled separately in playAudioQueue() when
        // the queue fully drains.
        // ──────────────────────────────────────────────────────────────

        // Stop any running reveal BEFORE final rendering.
        // Prevents stale timers, duplicate updates, or reveal loops
        // that survive past completion.
        this._stopSynchronizedReveal();

        // Mark response as complete — used by triple-condition reveal gate.
        this._responseComplete = true;
        this.metrics.responseComplete = performance.now();

        // Persist assistant response into the conversation that owns the stream,
        // even if the user switched to another chat before completion.
        if (this.currentAssistantResponse.trim() && this.streamingConversationId) {
            const owningConversation = this.conversations[this.streamingConversationId];

            if (owningConversation) {
                owningConversation.messages.push({
                    role: 'assistant',
                    content: this.currentAssistantResponse
                });

                // Persist the finalized assistant message to localStorage.
                this.persistConversations();
            }
        }

        // For voice turns where audio is playing:
        // Try to start synchronized reveal now that full text is available.
        // Reveal requires all 3 conditions: _responseComplete, _audioPlaybackStarted,
        // and _audioPlaybackPrepared. If any hasn't fired yet, the call from
        // playAudioQueue (on first scheduled buffer) will trigger it later.
        if (this._isVoiceTurn && this.isSpeaking) {
            this._tryStartSynchronizedReveal();
            // Don't render final text yet — let reveal finish naturally.
            // Full text will be shown when reveal completes.
            // Ownership cleanup is deferred to playAudioQueue drain.
        } else {
            // Non-voice turn or no audio: render full text immediately.
            if (this.streamingConversationId === this.activeConversationId) {
                this.updateTranscript('assistant', this.currentAssistantResponse, false);
            }
        }

        // Reset voice turn flag.
        const wasVoiceTurn = this._isVoiceTurn;
        this._isVoiceTurn = false;

        if (wasVoiceTurn) {
            this._awaitingPlaybackCleanup = true;
        }

        // For text turns (no audio), clean up ownership immediately.
        // For voice turns, ownership cleanup is deferred to playAudioQueue
        // drain or stopAudioPlayback — NOT done here.
        if (!wasVoiceTurn) {
            this.currentAssistantResponse = '';
            this.currentAssistantMessageEl = null;
            this.streamingConversationId = null;
            this.updateTelemetryMetric('metricStreamingState', 'Idle');

            const totalStart = this.metrics.sttStart ?? this.metrics.llmStart;
            if (totalStart !== null) {
                const totalLatency = performance.now() - totalStart;
                this.updateTelemetryMetric('metricTotalLatency', this._formatMs(totalLatency));
            }

            // TEXT-ONLY COMPLETION SAFETY:
            // If no audio playback ever started,
            // restore Ready state here.
            if (!this.isPlaying && !this.isSpeaking) {
                this.updateStatus('Ready', 'idle');
                this.setAppState('idle');
                this.updateMicText('idle');
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    // SYNCHRONIZED TEXT REVEAL SYSTEM
    // Progressive word-by-word reveal synced with REAL audio duration.
    // ══════════════════════════════════════════════════════════════

    /**
     * _tryStartSynchronizedReveal()
     * Triple-condition gate: only starts reveal when ALL are true:
     *   1. _responseComplete     — LLM/TTS generation finished (full text available)
     *   2. _audioPlaybackStarted — first audio chunk has arrived
     *   3. _audioPlaybackPrepared — first buffer is scheduled on audio graph
     *
     * NOTE: _responseComplete means generation completion, NOT playback
     * completion. Audio may still be buffered and playing.
     *
     * Called from:
     *   - handleResponseComplete() — when generation finishes
     *   - playAudioQueue() — when first buffer is scheduled
     * Whichever fires last triggers the reveal.
     */
    _tryStartSynchronizedReveal() {
        // All three conditions must be true
        if (!this._responseComplete) return;
        if (!this._audioPlaybackStarted) return;
        if (!this._audioPlaybackPrepared) return;

        // Don't restart if reveal is already running
        if (this._revealTimerId !== null) return;

        const fullText = this.currentAssistantResponse.trim();
        if (!fullText) return;

        this._startSynchronizedReveal(fullText, this._totalAudioDuration);
    }

    /**
     * _startSynchronizedReveal(fullText, audioDurationSec)
     * Begins progressively revealing text synchronized with REAL audio duration.
     *
     * FIX 1: Uses actual audioBuffer.duration accumulated during playback,
     * NOT estimated word timing. This accounts for punctuation pauses,
     * Hindi/English mix, and speaking style variations.
     *
     * @param {string} fullText - The complete response text to reveal
     * @param {number} audioDurationSec - REAL total audio duration in seconds
     */
    _startSynchronizedReveal(fullText, audioDurationSec) {
        // Stop any existing reveal
        this._stopSynchronizedReveal();

        if (!fullText) return;

        this._revealFullText = fullText;
        // Match word + trailing whitespace as single tokens.
        // This avoids whitespace-only reveal steps that cause
        // uneven pacing and inconsistent animation rhythm.
        this._revealWords = fullText.match(/\S+\s*/g) || [];
        this._revealIndex = 0;

        // Use REAL audio duration with a safety floor.
        // If duration is somehow 0 or very short, fall back to a minimum.
        const durationMs = Math.max(audioDurationSec * 1000, 2000);

        // Calculate reveal pacing:
        // Reveal in small word groups (~2-4 tokens per step) for smooth feel.
        const totalTokens = this._revealWords.length;
        const revealSteps = Math.max(10, Math.ceil(totalTokens / 3));
        const intervalMs = Math.max(50, durationMs / revealSteps);
        const tokensPerStep = Math.max(1, Math.ceil(totalTokens / revealSteps));

        console.log(
            `📝 Synchronized reveal: ${totalTokens} tokens, ` +
            `${audioDurationSec.toFixed(1)}s audio, ` +
            `${intervalMs.toFixed(0)}ms interval, ` +
            `${tokensPerStep} tokens/step`
        );

        this._revealTimerId = setInterval(() => {
            this._revealNextChunk(tokensPerStep);
        }, intervalMs);
    }

    /**
     * _revealNextChunk(count)
     * Reveals the next `count` tokens (words + whitespace) into the DOM.
     * Uses throttled scrolling to prevent layout thrashing.
     */
    _revealNextChunk(count) {
        if (this._revealIndex >= this._revealWords.length) {
            // All text revealed — stop timer and show final text.
            this._stopSynchronizedReveal();
            // Finalize the bubble as non-partial.
            if (this.currentAssistantMessageEl) {
                this.currentAssistantMessageEl.querySelector('.bubble').textContent =
                    this._revealFullText || this.currentAssistantResponse;
                this.currentAssistantMessageEl.classList.remove('partial');
                this.currentAssistantMessageEl.dataset.partial = 'false';
            }
            // NOTE: Ownership cleanup (currentAssistantResponse, streamingConversationId)
            // is NOT done here. Reveal completion does not guarantee playback completion.
            // Audio may still be draining. Cleanup happens in playAudioQueue() after
            // the queue fully drains, or in stopAudioPlayback() on interruption.
            return;
        }

        // Advance index
        const end = Math.min(this._revealIndex + count, this._revealWords.length);
        this._revealIndex = end;

        // Build revealed text so far
        const revealedText = this._revealWords.slice(0, this._revealIndex).join('');

        // Update the DOM (reuse existing partial bubble)
        if (this.currentAssistantMessageEl) {
            const bubble = this.currentAssistantMessageEl.querySelector('.bubble');
            if (bubble) {
                bubble.textContent = revealedText;
            }
        }

        // FIX 5: Throttled scrolling via requestAnimationFrame
        this._throttledScrollToBottom();
    }

    /**
     * _stopSynchronizedReveal()
     * Clears the reveal timer and resets all reveal state.
     * Called on:
     *   - reveal completion (all words shown)
     *   - barge-in / interruption (stopAudioPlayback)
     *   - conversation switch (via stopAudioPlayback)
     *   - response_complete cleanup
     */
    _stopSynchronizedReveal() {
        if (this._revealTimerId !== null) {
            clearInterval(this._revealTimerId);
            this._revealTimerId = null;
        }
        this._revealWords = [];
        this._revealIndex = 0;
        this._revealFullText = '';
        this._responseComplete = false;
        this._totalAudioDuration = 0;
    }

    /**
     * _throttledScrollToBottom()
     * FIX 5: Uses requestAnimationFrame to batch scroll updates,
     * preventing layout thrashing during rapid reveal ticks.
     */
    _throttledScrollToBottom() {
        if (this._scrollRafPending) return;
        this._scrollRafPending = true;
        requestAnimationFrame(() => {
            this.scrollToBottom();
            this._scrollRafPending = false;
        });
    }

    // ══════════════════════════════════════════════════════════════
    // TRANSCRIPT UI HELPERS (unchanged from original)
    // ══════════════════════════════════════════════════════════════

    updateTranscript(role, text, isPartial) {
        if (this.emptyStateEl) {
            this.emptyStateEl.classList.add('hidden');
        }

        const currentEl = role === 'user'
            ? this.currentUserMessageEl
            : this.currentAssistantMessageEl;

        if (isPartial) {
            if (currentEl && currentEl.dataset.partial === 'true') {
                currentEl.querySelector('.bubble').textContent = text;
                this.scrollToBottom();
                return;
            }

            const newEl = this.createMessageElement(role, text, true);
            this.transcriptEl.appendChild(newEl);
            if (role === 'user') {
                this.currentUserMessageEl = newEl;
            } else {
                this.currentAssistantMessageEl = newEl;
            }
            this.scrollToBottom();
            return;
        }

        if (currentEl && currentEl.dataset.partial === 'true') {
            currentEl.querySelector('.bubble').textContent = text;
            currentEl.classList.remove('partial');
            currentEl.dataset.partial = 'false';
            this.scrollToBottom();
            return;
        }

        const finalEl = this.createMessageElement(role, text, false);
        this.transcriptEl.appendChild(finalEl);
        if (role === 'user') {
            this.currentUserMessageEl = finalEl;
        } else {
            this.currentAssistantMessageEl = finalEl;
        }
        this.scrollToBottom();
    }

    updateStatus(message, className) {
        const normalized = className === 'success' ? 'idle' : className;
        if (this.modeBadge) {
            this.modeBadge.textContent = message;
            this.modeBadge.className = `status-pill ${normalized || ''}`.trim();
        }
        this.updateTelemetryMetric('metricLifecycle', message);
    }

    updateMicText(state) {
        if (!this.talkBtn) return;
        switch (state) {
            case 'listening':   this.talkBtn.textContent = 'Listening…'; break;
            case 'speaking':    this.talkBtn.textContent = 'Interrupt & speak'; break;
            case 'disconnected':this.talkBtn.textContent = 'Connecting…'; break;
            default:            this.talkBtn.textContent = 'Hold to speak';
        }
    }

    handlePointerDown(event) {
        if (this.talkBtn.disabled || this.isRecording) return;
        event.preventDefault();
        this.recordingStarted = false;
        this.holdTimerId = window.setTimeout(() => {
            this.holdTimerId = null;
            this.startRecording();
        }, this.holdThresholdMs);
    }

    handlePointerUp(event) {
        if (event) event.preventDefault();

        if (this.holdTimerId) {
            window.clearTimeout(this.holdTimerId);
            this.holdTimerId = null;
            if (!this.recordingStarted) {
                this.updateStatus('Press & hold to speak', 'idle');
                this.updateMicText('idle');
                return;
            }
        }

        if (this.isRecording) {
            this.stopRecording();
        }
    }

    updateConnectionStatus(message, isConnected) {
        if (this.connectionText) this.connectionText.textContent = message;
        if (this.connectionDot) this.connectionDot.classList.toggle('online', Boolean(isConnected));
        this.updateTelemetryMetric('metricWsStatus', message);
    }

    setAppState(state) {
        this.appEl.classList.remove(
            'state-idle', 'state-listening', 'state-processing', 'state-speaking'
        );
        this.appEl.classList.add(`state-${state}`);
    }

    createMessageElement(role, text, isPartial) {
        const item = document.createElement('div');
        item.className = `message ${role}`;
        if (isPartial) item.classList.add('partial');
        item.dataset.partial = isPartial ? 'true' : 'false';

        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = role === 'user' ? 'You' : 'Krishna';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = text;

        item.appendChild(label);
        item.appendChild(bubble);
        return item;
    }

    scrollToBottom() {
        this.transcriptEl.scrollTo({
            top: this.transcriptEl.scrollHeight,
            behavior: 'auto'
        });
    }

    // ══════════════════════════════════════════════════════════════
    // CHAT MESSAGE (text input)
    // [NEW] _saveMessage() added for user text + response persistence
    // ══════════════════════════════════════════════════════════════

    sendChatMessage() {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        const text = this.textInput.value.trim();
        if (!text) return;

        // FIX: Ensure a conversation always exists before sending
        if (!this.activeConversationId) {
            this.createNewConversation();
        }

        this.textInput.value = '';

        // Render immediately
        this.updateTranscript('user', text, false);

        // Persist user text message
        this._saveMessage('user', text);

        // FIX: Lock the assistant stream to this conversation
        this.streamingConversationId = this.activeConversationId;

        // FIX 3: Text mode — mark as NOT a voice turn so tokens render live.
        this._isVoiceTurn = false;
        this._responseComplete = false;
        this._totalAudioDuration = 0;

        this._resetTelemetryForNewResponse(false);
        this.metrics.llmStart = performance.now();
        this.updateTelemetryMetric('metricStreamingState', 'Text');
        this.updateTelemetryMetric('metricPlayback', 'Idle');

        this.currentAssistantResponse = '';
        this.currentAssistantMessageEl = null;
        this.updateStatus('Krishna is responding...', 'processing');

        this.ws.send(JSON.stringify({
            type: 'chat_message',
            text,
            conversation_id: this.activeConversationId,   // backend routing
        }));
    }

    // ══════════════════════════════════════════════════════════════
    // METRICS (unchanged)
    // ══════════════════════════════════════════════════════════════

    _initTelemetryPanel() {
        this.updateTelemetryMetric('metricProvider', 'Groq');
        this.updateTelemetryMetric('metricVoiceProvider', 'ElevenLabs');
        this.updateTelemetryMetric('metricWsStatus', this.connectionText?.textContent || 'Connecting');
        this.updateTelemetryMetric('metricLifecycle', this.modeBadge?.textContent || 'Idle');
        this.updateTelemetryMetric('metricPlayback', 'Idle');
        this.updateTelemetryMetric('metricAudioQueue', '0');
        this.updateTelemetryMetric('metricStreamingState', 'Idle');
        this._updateActiveConversationTelemetry();
    }

    _resetTelemetryForNewResponse(preserveSttStart) {
        if (!preserveSttStart) {
            this.metrics.sttStart = null;
        }
        this.metrics.transcriptReady = null;
        this.metrics.llmStart = null;
        this.metrics.firstToken = null;
        this.metrics.ttsStart = null;
        this.metrics.playbackStart = null;
        this.metrics.responseComplete = null;
        this.metrics.tokenCount = 0;
        this.metrics.lastTokenAt = null;

        this.updateTelemetryMetric('metricSttLatency', '--');
        this.updateTelemetryMetric('metricLlmLatency', '--');
        this.updateTelemetryMetric('metricTtsLatency', '--');
        this.updateTelemetryMetric('metricTotalLatency', '--');
        this.updateTelemetryMetric('metricTokensPerSec', '--');
    }

    _updateActiveConversationTelemetry() {
        const conv = this.conversations[this.activeConversationId];
        const title = conv ? conv.title : 'New conversation';
        this.updateTelemetryMetric('metricActiveConversation', title);
    }

    updateTelemetryMetric(id, value) {
        if (!id) return;
        const cached = this._telemetryCache[id];
        const el = cached || document.getElementById(id);
        if (!el) return;
        this._telemetryCache[id] = el;
        el.textContent = value;
    }

    _formatMs(value) {
        if (value === null || value === undefined || Number.isNaN(value)) return '--';
        return `${Math.round(value)}ms`;
    }

    _updateTokensPerSec(now) {
        if (!this.metrics.firstToken || this.metrics.tokenCount === 0) return;
        const elapsedSec = (now - this.metrics.firstToken) / 1000;
        if (elapsedSec <= 0) return;
        const tps = this.metrics.tokenCount / elapsedSec;
        this.updateTelemetryMetric('metricTokensPerSec', `${tps.toFixed(1)}`);
    }

    updateMetrics() {
        if (this.metrics.sttStart !== null && this.metrics.transcriptReady !== null) {
            const sttLatency = this.metrics.transcriptReady - this.metrics.sttStart;
            this.updateTelemetryMetric('metricSttLatency', this._formatMs(sttLatency));
        }

        if (this.metrics.llmStart !== null && this.metrics.firstToken !== null) {
            const llmLatency = this.metrics.firstToken - this.metrics.llmStart;
            this.updateTelemetryMetric('metricLlmLatency', this._formatMs(llmLatency));
        }

        if (this.metrics.ttsStart !== null) {
            const ttsBase = this.metrics.firstToken ?? this.metrics.llmStart;
            if (ttsBase !== null) {
                const ttsLatency = this.metrics.ttsStart - ttsBase;
                this.updateTelemetryMetric('metricTtsLatency', this._formatMs(ttsLatency));
            }
        }
    }

    getLatencyClass(latency, target) {
        if (latency < target * 0.8) return 'latency-good';
        if (latency < target * 1.2) return 'latency-ok';
        return 'latency-bad';
    }
}

// Initialize on page load
window.addEventListener('load', () => {
    new StreamingVoiceClient();
});