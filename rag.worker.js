import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';
import * as webllm from 'https://esm.run/@mlc-ai/web-llm';

const CHANNEL = 'FLYRAG_CORE';
const encoder = new TextEncoder();
const MIN_COARSE_CANDIDATES = 8;
const EMBEDDING_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const GENERATION_MODEL_ID = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';

const state = {
  quiet: false,
  initialized: false,
  wasm: null,
  memory: null,
  queryPtr: 0,
  candidatePtr: 0,
  bufferCapacity: 0,
  embedder: null,
  llm: null,
  chunks: [],
  initializing: null,
  initializingPayloadKey: null,
};

function post(type, payload = {}) {
  self.postMessage({ channel: CHANNEL, type, ...payload });
}

function log(...args) {
  if (!state.quiet) {
    console.log('[FlyRAGWorker]', ...args);
  }
}

function toPercent(ratio, min, max) {
  const clamped = Math.max(0, Math.min(1, ratio));
  return Math.round(min + clamped * (max - min));
}

async function initWasm(wasmUrl = './assembly/flyhash.wasm') {
  const response = await fetch(wasmUrl);
  const bytes = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {
    env: {
      abort(message, fileName, lineNumber, columnNumber) {
        throw new Error(
          `Wasm abort at ${lineNumber}:${columnNumber} (messagePtr=${message}, filePtr=${fileName})`,
        );
      },
    },
  });
  const exports = instance.exports;

  state.wasm = exports;
  state.memory = exports.memory;
  state.queryPtr = Number(exports.queryBufferPtr());
  state.candidatePtr = Number(exports.candidateBufferPtr());
  state.bufferCapacity = Number(exports.bufferCapacity());
  return exports;
}

function writeText(ptr, text) {
  const rawBytes = encoder.encode(text ?? '');
  const length = Math.min(rawBytes.length, state.bufferCapacity);
  const view = new Uint8Array(state.memory.buffer, ptr, state.bufferCapacity);
  view.fill(0);
  view.set(rawBytes.subarray(0, length));
  return length;
}

function flyScore(queryText, candidateText) {
  const qLen = writeText(state.queryPtr, queryText);
  const cLen = writeText(state.candidatePtr, candidateText);
  return Number(state.wasm.scoreFromBuffers(qLen, cLen));
}

function cosine(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function toFiniteInt(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.trunc(parsed);
}

async function embedText(text) {
  const result = await state.embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(result.data);
}

function chunkText(source, size = 600, overlap = 100) {
  const text = (source ?? '').trim();
  if (!text) return [];

  const chunks = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + size, text.length);
    const chunk = text.slice(start, end).trim();
    if (chunk) chunks.push(chunk);

    if (end >= text.length) break;
    start = Math.max(end - overlap, start + 1);
  }

  return chunks;
}

function rankByDenseSimilarity(queryEmbedding, candidates, topK) {
  return candidates
    .map((item) => ({ ...item, denseScore: cosine(queryEmbedding, item.embedding) }))
    .sort((a, b) => b.denseScore - a.denseScore)
    .slice(0, topK);
}

async function initialize(payload) {
  state.quiet = Boolean(payload?.quiet);
  try {
    post('PROGRESS', { percent: 1 });

    const wasmExports = await initWasm(payload?.wasmUrl);
    if (payload?.flyHashConfig) {
      const cfg = payload.flyHashConfig;
      wasmExports.configure(
        toFiniteInt(cfg.hashBits, 512),
        toFiniteInt(cfg.winners, 48),
        toFiniteInt(cfg.projections, 6),
        toFiniteInt(cfg.seed, 0xC0FFEE) >>> 0,
      );
    }
    post('PROGRESS', { percent: 12 });

    state.embedder = await pipeline('feature-extraction', EMBEDDING_MODEL_ID, {
      progress_callback(progress) {
        if (progress?.total == null || progress?.loaded == null || progress.total <= 0) return;
        const ratio = progress.loaded / progress.total;
        post('PROGRESS', { percent: toPercent(ratio, 12, 72) });
      },
    });

    post('PROGRESS', { percent: 74 });

    state.llm = await webllm.CreateMLCEngine(GENERATION_MODEL_ID, {
      initProgressCallback(progress) {
        const ratio = typeof progress?.progress === 'number' ? progress.progress : 0;
        post('PROGRESS', { percent: toPercent(ratio, 74, 100) });
      },
    });

    state.initialized = true;
    post('PROGRESS', { percent: 100 });
  } catch (error) {
    state.initialized = false;
    state.initializing = null;
    state.initializingPayloadKey = null;
    state.wasm = null;
    state.memory = null;
    state.queryPtr = 0;
    state.candidatePtr = 0;
    state.bufferCapacity = 0;
    state.embedder = null;
    state.llm = null;
    state.chunks = [];
    throw error;
  }
}

async function loadDocuments(payload) {
  const text = payload?.text ?? '';
  const chunks = chunkText(text);

  const embeddedChunks = [];
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const embedding = await embedText(chunk);
    embeddedChunks.push({ id: i, text: chunk, embedding });
  }

  state.chunks = embeddedChunks;
  log(`Loaded ${embeddedChunks.length} chunks`);
  return { chunkCount: embeddedChunks.length };
}

async function ask(payload, requestId) {
  const question = (payload?.question ?? '').trim();
  if (!question) {
    throw new Error('Question must be non-empty.');
  }

  if (!state.chunks.length) {
    throw new Error('No document chunks loaded. Call LOAD_DOCUMENTS first.');
  }

  const topK = Math.max(1, toFiniteInt(payload?.topK, 4));
  const coarseLimit = Math.min(state.chunks.length, Math.max(topK * 4, MIN_COARSE_CANDIDATES));

  const coarse = state.chunks
    .map((chunk) => ({ ...chunk, flyScore: flyScore(question, chunk.text) }))
    .sort((a, b) => b.flyScore - a.flyScore)
    .slice(0, coarseLimit);

  const queryEmbedding = await embedText(question);
  const reranked = rankByDenseSimilarity(queryEmbedding, coarse, topK);

  const context = reranked
    .map((item, idx) => `Context ${idx + 1}:\n${item.text}`)
    .join('\n\n');

  const messages = [
    {
      role: 'system',
      content:
        'You are a retrieval-augmented assistant. Answer the question only using provided context. If context is insufficient, say so clearly.',
    },
    {
      role: 'user',
      content: `Question:\n${question}\n\nRetrieved context:\n${context}`,
    },
  ];

  let answer = '';
  const stream = await state.llm.chat.completions.create({
    messages,
    temperature: 0.2,
    max_tokens: 512,
    stream: true,
  });

  for await (const part of stream) {
    const token = part?.choices?.[0]?.delta?.content ?? '';
    if (!token) continue;

    answer += token;
    post('STREAM', { requestId, token });
  }

  return {
    answer,
    contexts: reranked.map((item) => ({ text: item.text, flyScore: item.flyScore, denseScore: item.denseScore })),
  };
}

// Custom message router: only messages on our channel are handled by this worker
// so external handlers used by model libraries cannot intercept command traffic.
self.addEventListener('message', async (event) => {
  const message = event.data;
  if (!message || message.channel !== CHANNEL) return;

  const { type, payload = {}, requestId } = message;

  try {
    if (type === 'INIT') {
      const payloadKey = JSON.stringify(payload ?? {});
      if (!state.initialized) {
        if (!state.initializing) {
          state.initializingPayloadKey = payloadKey;
          state.initializing = initialize(payload);
        } else if (state.initializingPayloadKey !== payloadKey) {
          throw new Error('INIT already in progress with different configuration payload.');
        }

        const initPromise = state.initializing;
        try {
          await initPromise;
        } finally {
          if (state.initializing === initPromise) {
            state.initializing = null;
            state.initializingPayloadKey = null;
          }
        }
      }
      post('READY', { requestId });
      return;
    }

    if (!state.initialized) {
      throw new Error('Worker is not initialized. Call INIT first.');
    }

    if (type === 'LOAD_DOCUMENTS') {
      const result = await loadDocuments(payload);
      post('RESULT', { requestId, result });
      return;
    }

    if (type === 'ASK') {
      const result = await ask(payload, requestId);
      post('RESULT', { requestId, result });
      return;
    }

    if (type === 'RESET') {
      state.chunks = [];
      post('RESULT', { requestId, result: { ok: true } });
      return;
    }

    throw new Error(`Unknown command type: ${type}`);
  } catch (error) {
    post('ERROR', {
      requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
