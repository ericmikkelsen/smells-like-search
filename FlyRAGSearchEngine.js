const CHANNEL = 'FLYRAG_CORE';

export class FlyRAGSearchEngine {
  constructor({ quiet = false, workerUrl = './rag.worker.js', onProgress, wasmUrl, flyHashConfig } = {}) {
    this.quiet = Boolean(quiet);
    this.onProgress = typeof onProgress === 'function' ? onProgress : null;
    this.worker = new Worker(workerUrl, { type: 'module' });

    this.requestId = 1;
    this.pending = new Map();
    this.streamHandlers = new Map();
    this.isInitializing = false;
    this.initFailed = false;

    this.worker.addEventListener('message', (event) => this.#handleMessage(event.data));
    this.worker.addEventListener('error', (error) => {
      if (!this.quiet) console.error('[FlyRAGSearchEngine] Worker error:', error);
    });

    this.initPayload = { quiet: this.quiet, wasmUrl, flyHashConfig };
    this.ready = this.#startInit();
  }

  async initialize({ retry = this.initFailed } = {}) {
    if (retry && this.initFailed) {
      await this.#startInit();
      return;
    }
    await this.ready;
  }

  async loadDocuments(text) {
    await this.ready;
    const result = await this.#send('LOAD_DOCUMENTS', { text: text ?? '' });
    return result.chunkCount;
  }

  async ask(question, { topK = 4, onToken } = {}) {
    await this.ready;

    const requestId = this.#nextRequestId();
    if (typeof onToken === 'function') {
      this.streamHandlers.set(requestId, onToken);
    }

    try {
      return await this.#sendWithId(requestId, 'ASK', { question, topK });
    } finally {
      this.streamHandlers.delete(requestId);
    }
  }

  async reset() {
    await this.ready;
    await this.#send('RESET', {});
  }

  dispose() {
    this.pending.forEach(({ reject }) => reject(new Error('FlyRAGSearchEngine disposed.')));
    this.pending.clear();
    this.streamHandlers.clear();
    this.worker.terminate();
  }

  #send(type, payload) {
    return this.#sendWithId(this.#nextRequestId(), type, payload);
  }

  #startInit() {
    if (this.isInitializing) {
      return this.ready;
    }

    this.isInitializing = true;
    const initPromise = this.#send('INIT', this.initPayload)
      .then((result) => {
        this.initFailed = false;
        return result;
      })
      .catch((error) => {
        this.initFailed = true;
        throw error;
      })
      .finally(() => {
        this.isInitializing = false;
      });
    this.ready = initPromise;
    return initPromise;
  }

  #nextRequestId() {
    const id = this.requestId;
    this.requestId += 1;
    return id;
  }

  #sendWithId(requestId, type, payload) {
    const promise = new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
    });

    this.worker.postMessage({
      channel: CHANNEL,
      type,
      requestId,
      payload,
    });

    return promise;
  }

  #handleMessage(message) {
    if (!message || message.channel !== CHANNEL) return;

    if (message.type === 'PROGRESS') {
      if (this.onProgress) this.onProgress({ type: 'PROGRESS', percent: Number(message.percent ?? 0) });
      return;
    }

    if (message.type === 'STREAM') {
      const onToken = this.streamHandlers.get(message.requestId);
      if (onToken) onToken(message.token ?? '');
      return;
    }

    const pending = this.pending.get(message.requestId);
    if (!pending) return;

    this.pending.delete(message.requestId);

    if (message.type === 'ERROR') {
      pending.reject(new Error(message.message || 'Unknown worker error'));
      return;
    }

    if (message.type === 'READY') {
      pending.resolve({ ok: true });
      return;
    }

    if (message.type === 'RESULT') {
      pending.resolve(message.result);
      return;
    }

    pending.reject(new Error(`Unknown worker response type: ${message.type}`));
  }
}

export default FlyRAGSearchEngine;
