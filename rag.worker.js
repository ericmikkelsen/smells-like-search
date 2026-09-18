import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';
import * as webllm from 'https://esm.run/@mlc-ai/web-llm';

const CHANNEL = 'FLYRAG_CORE';
const encoder = new TextEncoder();
const MIN_COARSE_CANDIDATES = 8;
const MAX_ASK_ATTEMPTS = 3;
const TOPK_DEFAULT_SCHEDULE = [4, 8, 12];
const EMBEDDING_MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const GENERATION_MODEL_ID = 'Llama-3.2-3B-Instruct-q4f16_1-MLC';
const RAG_SYSTEM_PROMPT =
  'You are a retrieval-augmented assistant. Answer the question only using provided context. If context is insufficient, say so clearly.';

// Generation backend: 'nano' = Chrome Prompt API, 'webllm' = WebLLM/Llama
const LLM_BACKEND = {
  NANO: 'nano',
  WEBLLM: 'webllm',
};

const state = {
  quiet: false,
  initialized: false,
  wasm: null,
  memory: null,
  queryPtr: 0,
  candidatePtr: 0,
  candidateBitsPtr: 0,
  bufferCapacity: 0,
  hashBitWords: 0,
  embedder: null,
  llm: null,
  llmBackend: null,
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
  state.candidateBitsPtr = Number(exports.candidateBitsPtr());
  state.bufferCapacity = Number(exports.bufferCapacity());
  state.hashBitWords = Number(exports.hashBitWords());
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

// Pre-compute a fly hash for a piece of text. Returns a Uint32Array copy of the bit vector.
function flyHash(text) {
  const cLen = writeText(state.candidatePtr, text);
  state.wasm.hashCandidateBuffer(cLen);
  const words = state.hashBitWords;
  const bits = new Uint32Array(state.memory.buffer, state.candidateBitsPtr, words);
  return bits.slice(); // copy out of WASM memory
}

// Score the query (already hashed into queryBits) against a pre-computed bit vector.
function flyScorePrecomputed(storedBits) {
  const words = state.hashBitWords;
  const dest = new Uint32Array(state.memory.buffer, state.candidateBitsPtr, words);
  dest.set(storedBits);
  return Number(state.wasm.overlapPrecomputed());
}

// Hash query text into queryBits once, then score it against every chunk's stored bits.
function flyScoreAllChunks(queryText, chunks) {
  const qLen = writeText(state.queryPtr, queryText);
  state.wasm.hashQueryBuffer(qLen);
  return chunks.map((chunk) => ({ ...chunk, flyScore: flyScorePrecomputed(chunk.flyBits) }));
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

function tokenize(text) {
  return String(text ?? '')
    .toLowerCase()
    .match(/[a-z0-9]{3,}/g) ?? [];
}

const STOP_WORDS = new Set([
  'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 'was', 'were', 'have', 'has', 'had', 'not', 'but', 'you',
  'your', 'can', 'could', 'would', 'should', 'about', 'there', 'their', 'them', 'they', 'his', 'her', 'she', 'him', 'our',
  'out', 'what', 'when', 'where', 'which', 'who', 'how', 'why', 'all', 'any', 'only', 'into', 'then', 'than', 'also', 'more',
  'very', 'some', 'just', 'like', 'one', 'two', 'three', 'text', 'context', 'provided',
]);

function toKeywordSet(text) {
  const tokens = tokenize(text).filter((token) => !STOP_WORDS.has(token));
  return new Set(tokens);
}

function ratioCovered(sourceTokens, evidenceSet) {
  if (!sourceTokens.length) return 0;
  let hits = 0;
  for (const token of sourceTokens) {
    if (evidenceSet.has(token)) hits += 1;
  }
  return hits / sourceTokens.length;
}

function hasInsufficientCue(answer) {
  return /(insufficient|not enough|cannot answer|can't answer|cannot determine|can't determine|do not have enough|don't have enough)/i.test(answer);
}

function extractCapitalizedWords(text) {
  const matches = String(text ?? '').match(/\b[A-Z][a-z]{2,}\b/g) ?? [];
  return matches.map((item) => item.toLowerCase());
}

function buildTopKSchedule(baseTopK) {
  const first = Math.max(1, toFiniteInt(baseTopK, TOPK_DEFAULT_SCHEDULE[0]));
  return [
    first,
    Math.max(TOPK_DEFAULT_SCHEDULE[1], first * 2),
    Math.max(TOPK_DEFAULT_SCHEDULE[2], first * 3),
  ].slice(0, MAX_ASK_ATTEMPTS);
}

function rewriteQuestionForRecall(question) {
  const tokens = Array.from(toKeywordSet(question)).slice(0, 8);
  if (!tokens.length) return question;
  return `${question}\nRelated terms: ${tokens.join(', ')}`;
}

async function retrieveContexts(query, topK, attemptIndex) {
  const coarseMultiplier = Math.min(8, 4 + attemptIndex * 2);
  const coarseFloor = MIN_COARSE_CANDIDATES + attemptIndex * 4;
  const coarseLimit = Math.min(state.chunks.length, Math.max(topK * coarseMultiplier, coarseFloor));

  const coarse = flyScoreAllChunks(query, state.chunks)
    .sort((a, b) => b.flyScore - a.flyScore)
    .slice(0, coarseLimit);

  const queryEmbedding = await embedText(query);
  const reranked = rankByDenseSimilarity(queryEmbedding, coarse, topK);
  return { reranked, coarseLimit };
}

function buildContextString(reranked) {
  return reranked
    .map((item, idx) => `Context ${idx + 1} (chunk ${item.id}):\n${item.text}`)
    .join('\n\n');
}

function evaluateAnswerQuality({ answer, question, contexts, attemptIndex }) {
  const answerKeywords = Array.from(toKeywordSet(answer));
  const questionKeywords = Array.from(toKeywordSet(question));
  const contextKeywordSet = toKeywordSet(contexts.map((item) => item.text).join('\n'));
  const questionKeywordSet = new Set(questionKeywords);

  const groundedness = ratioCovered(answerKeywords, contextKeywordSet);
  const relevance = ratioCovered(answerKeywords, questionKeywordSet);

  const namedEntities = extractCapitalizedWords(answer);
  const contextCapitalized = new Set(extractCapitalizedWords(contexts.map((item) => item.text).join('\n')));
  const questionCapitalized = new Set(extractCapitalizedWords(question));
  const unsupportedEntities = namedEntities.filter(
    (token) => !contextCapitalized.has(token) && !questionCapitalized.has(token),
  );
  const contradictionRisk = namedEntities.length ? unsupportedEntities.length / namedEntities.length : 0;
  const contradictionScore = 1 - contradictionRisk;

  const insufficientCue = hasInsufficientCue(answer);
  const uncertainOrLowGrounding = groundedness < 0.72;
  const uncertaintyHandled = !uncertainOrLowGrounding || insufficientCue;

  const lowEvidenceSafeRefusal = insufficientCue && contradictionScore >= 0.7 && answer.length <= 360;
  const passesGrounding = groundedness >= 0.72 && contradictionScore >= 0.7 && relevance >= 0.15;

  const pass = passesGrounding || lowEvidenceSafeRefusal;
  let failReason = '';
  if (!pass) {
    if (!uncertaintyHandled) failReason = 'missing_uncertainty';
    else if (contradictionScore < 0.7) failReason = 'possible_hallucination';
    else if (groundedness < 0.72) failReason = 'low_groundedness';
    else if (relevance < 0.15) failReason = 'low_relevance';
    else failReason = 'quality_gate_failed';
  }

  return {
    pass,
    failReason,
    stopCandidate: pass ? 'quality_passed' : '',
    scores: {
      groundedness: Number(groundedness.toFixed(3)),
      relevance: Number(relevance.toFixed(3)),
      contradiction: Number(contradictionScore.toFixed(3)),
      uncertaintyHandled,
    },
    details: {
      attempt: attemptIndex + 1,
      insufficientCue,
      unsupportedEntities,
    },
  };
}

async function generateAnswer(question, context) {
  if (state.llmBackend === LLM_BACKEND.NANO) {
    let session;
    try {
      session = await self.ai.languageModel.create({
        initialPrompts: [{ role: 'system', content: RAG_SYSTEM_PROMPT }],
      });
    } catch (nanoError) {
      throw new Error(`Gemini Nano session creation failed: ${nanoError instanceof Error ? nanoError.message : String(nanoError)}`);
    }
    try {
      const prompt = `Question:\n${question}\n\nRetrieved context:\n${context}`;
      return await session.prompt(prompt);
    } catch (promptError) {
      throw new Error(`Gemini Nano generation failed: ${promptError instanceof Error ? promptError.message : String(promptError)}`);
    } finally {
      session.destroy();
    }
  }

  if (!state.llm) {
    throw new Error('Language model engine is not available.');
  }

  const completion = await state.llm.chat.completions.create({
    messages: [
      { role: 'system', content: RAG_SYSTEM_PROMPT },
      { role: 'user', content: `Question:\n${question}\n\nRetrieved context:\n${context}` },
    ],
    temperature: 0.2,
    max_tokens: 512,
  });
  return completion?.choices?.[0]?.message?.content ?? '';
}

async function checkNanoAvailable() {
  try {
    if (typeof self.ai?.languageModel?.capabilities !== 'function') return false;
    const caps = await self.ai.languageModel.capabilities();
    return caps?.available === 'readily';
  } catch {
    return false;
  }
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
      // Re-read hashBitWords after configure() since activeHashBits may have changed.
      state.hashBitWords = Number(wasmExports.hashBitWords());
    }
    post('PROGRESS', { percent: 12 });

    post('STATUS', { text: 'Downloading embedding model…' });
    state.embedder = await pipeline('feature-extraction', EMBEDDING_MODEL_ID, {
      dtype: 'q8',
      progress_callback(progress) {
        if (progress?.loaded == null) return;
        if (progress.total != null && progress.total > 0) {
          const ratio = progress.loaded / progress.total;
          post('PROGRESS', { percent: toPercent(ratio, 12, 72) });
        } else if (progress.loaded > 0) {
          // Content-Length header unavailable — pulse progress so the UI doesn't appear stuck.
          const MB = progress.loaded / (1024 * 1024);
          // Asymptotically approach 70% as bytes accumulate (saturates around 100 MB).
          const ratio = 1 - Math.exp(-MB / 30);
          post('PROGRESS', { percent: toPercent(ratio, 12, 72) });
        }
      },
    });

    post('PROGRESS', { percent: 74 });

    // Prefer Chrome's built-in Gemini Nano (Prompt API) if available.
    const nanoAvailable = await checkNanoAvailable();
    if (nanoAvailable) {
      // No session needed at init time; a fresh session is created per-question.
      state.llmBackend = LLM_BACKEND.NANO;
      post('PROGRESS', { percent: 100 });
      post('STATUS', { text: 'Ready (Gemini Nano).' });
    } else {
      post('STATUS', { text: 'Downloading language model…' });
      state.llm = await webllm.CreateMLCEngine(GENERATION_MODEL_ID, {
        initProgressCallback(progress) {
          const ratio = typeof progress?.progress === 'number' ? progress.progress : 0;
          post('PROGRESS', { percent: toPercent(ratio, 74, 100) });
          if (progress?.text) post('STATUS', { text: progress.text });
        },
      });
      state.llmBackend = LLM_BACKEND.WEBLLM;
      post('PROGRESS', { percent: 100 });
    }

    state.initialized = true;
  } catch (error) {
    state.initialized = false;
    state.initializing = null;
    state.initializingPayloadKey = null;
    state.wasm = null;
    state.memory = null;
    state.queryPtr = 0;
    state.candidatePtr = 0;
    state.candidateBitsPtr = 0;
    state.bufferCapacity = 0;
    state.hashBitWords = 0;
    state.embedder = null;
    state.llm = null;
    state.llmBackend = null;
    state.chunks = [];
    throw error;
  }
}

async function loadDocuments(payload) {
  const text = payload?.text ?? '';
  const chunks = chunkText(text);
  const total = chunks.length;

  post('STATUS', { text: `Embedding chunk 0 of ${total}…` });

  const embeddedChunks = [];
  for (let i = 0; i < total; i++) {
    const chunk = chunks[i];
    const embedding = await embedText(chunk);
    const flyBits = flyHash(chunk);
    embeddedChunks.push({ id: i, text: chunk, embedding, flyBits });
    if ((i + 1) % 5 === 0 || i + 1 === total) {
      post('STATUS', { text: `Embedding chunk ${i + 1} of ${total}…` });
    }
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

  const topKSchedule = buildTopKSchedule(payload?.topK);
  const attemptDebug = [];
  const seenContextIds = new Set();
  let finalAnswer = '';
  let finalContexts = [];
  let finalQuality = null;
  let stopReason = 'max_attempts_reached';

  for (let attemptIndex = 0; attemptIndex < MAX_ASK_ATTEMPTS; attemptIndex++) {
    const topK = topKSchedule[Math.min(attemptIndex, topKSchedule.length - 1)];
    const retrievalQuestion = attemptIndex === MAX_ASK_ATTEMPTS - 1 ? rewriteQuestionForRecall(question) : question;
    const { reranked, coarseLimit } = await retrieveContexts(retrievalQuestion, topK, attemptIndex);
    const context = buildContextString(reranked);
    const contextIds = reranked.map((item) => item.id);

    const newlyAddedCount = contextIds.filter((id) => !seenContextIds.has(id)).length;
    for (const id of contextIds) {
      seenContextIds.add(id);
    }

    const answer = await generateAnswer(question, context);
    const quality = evaluateAnswerQuality({
      answer,
      question,
      contexts: reranked,
      attemptIndex,
    });

    attemptDebug.push({
      attempt: attemptIndex + 1,
      retrievalQuestion,
      topK,
      coarseLimit,
      contextIds,
      newlyAddedContextCount: newlyAddedCount,
      quality,
    });

    finalAnswer = answer;
    finalContexts = reranked;
    finalQuality = quality;

    if (quality.pass) {
      stopReason = 'quality_passed';
      break;
    }

    const retrievalSaturated = attemptIndex > 0 && newlyAddedCount === 0;
    const lastAttempt = attemptIndex === MAX_ASK_ATTEMPTS - 1;
    if (retrievalSaturated) {
      stopReason = 'retrieval_saturated';
      break;
    }
    if (lastAttempt) {
      stopReason = 'low_confidence_after_expansion';
      break;
    }
  }

  if (!finalQuality?.pass) {
    finalAnswer = `I don’t have enough grounded context to answer that reliably. ${finalQuality?.failReason ? `Reason: ${finalQuality.failReason}.` : ''} Please provide more source text or a more specific excerpt.`;
  }

  if (finalAnswer) {
    post('STREAM', { requestId, token: finalAnswer });
  }

  return {
    answer: finalAnswer,
    contexts: finalContexts.map((item) => ({
      id: item.id,
      text: item.text,
      flyScore: item.flyScore,
      denseScore: item.denseScore,
    })),
    debug: {
      attempts: attemptDebug,
      stopReason,
      passedQualityGate: Boolean(finalQuality?.pass),
      finalScores: finalQuality?.scores ?? null,
    },
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
