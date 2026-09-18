// Lightweight Fruit Fly-style sparse hashing for coarse retrieval.
// This module is designed for AssemblyScript -> Wasm compilation.

const FEATURE_DIM: i32 = 256;
const MAX_HASH_BITS: i32 = 2048;
const MAX_WORDS: i32 = MAX_HASH_BITS / 32;
const MAX_WINNERS: i32 = 256;
const MAX_PROJECTIONS: i32 = 16;
const MAX_INPUT_BYTES: i32 = 8192;

let activeHashBits: i32 = 512;
let activeWinners: i32 = 48;
let activeProjections: i32 = 6;
let activeSeed: u32 = 0xC0FFEE;

const queryBuffer = new StaticArray<u8>(MAX_INPUT_BYTES);
const candidateBuffer = new StaticArray<u8>(MAX_INPUT_BYTES);
const queryPtr = changetype<usize>(queryBuffer);
const candidatePtr = changetype<usize>(candidateBuffer);

const queryFeatures = new StaticArray<f32>(FEATURE_DIM);
const candidateFeatures = new StaticArray<f32>(FEATURE_DIM);

const queryBits = new StaticArray<u32>(MAX_WORDS);
const candidateBits = new StaticArray<u32>(MAX_WORDS);

// Scratch buffers reused per call for speed. This module assumes single-threaded Wasm execution.
const topScores = new StaticArray<f32>(MAX_WINNERS);
const topIndices = new StaticArray<i32>(MAX_WINNERS);

@inline
function hash32(value: u32): u32 {
  let x = value;
  x ^= x >> 16;
  x *= 0x7FEB352D;
  x ^= x >> 15;
  x *= 0x846CA68B;
  x ^= x >> 16;
  return x;
}

@inline
function clearFeatures(features: StaticArray<f32>): void {
  for (let i: i32 = 0; i < FEATURE_DIM; i++) {
    unchecked(features[i] = 0.0);
  }
}

@inline
function clearBits(bits: StaticArray<u32>, words: i32): void {
  for (let i: i32 = 0; i < words; i++) {
    unchecked(bits[i] = 0);
  }
}

@inline
function minI32(a: i32, b: i32): i32 {
  return a < b ? a : b;
}

function buildFeatures(ptr: usize, byteLength: i32, outFeatures: StaticArray<f32>): void {
  clearFeatures(outFeatures);

  let cappedLength = minI32(byteLength, MAX_INPUT_BYTES);
  // Fast length normalization for coarse retrieval; exact dense normalization happens in reranking.
  let invNorm: f32 = cappedLength > 0 ? 1.0 / Mathf.sqrt(<f32>cappedLength) : 1.0;

  for (let i: i32 = 0; i < cappedLength; i++) {
    let featureIdx = <i32>load<u8>(ptr + <usize>i);
    unchecked(outFeatures[featureIdx] += invNorm);
  }
}

function insertTop(score: f32, index: i32, winners: i32): void {
  let boundedWinners = minI32(winners, MAX_WINNERS);
  if (boundedWinners <= 0) return;
  if (score < unchecked(topScores[boundedWinners - 1])) return;

  unchecked(topScores[boundedWinners - 1] = score);
  unchecked(topIndices[boundedWinners - 1] = index);

  for (let i: i32 = boundedWinners - 1; i > 0; i--) {
    let left = i - 1;
    if (unchecked(topScores[i]) <= unchecked(topScores[left])) break;

    let tmpScore = unchecked(topScores[left]);
    let tmpIdx = unchecked(topIndices[left]);
    unchecked(topScores[left] = topScores[i]);
    unchecked(topIndices[left] = topIndices[i]);
    unchecked(topScores[i] = tmpScore);
    unchecked(topIndices[i] = tmpIdx);
  }
}

function hashFeatures(features: StaticArray<f32>, outBits: StaticArray<u32>): void {
  let words = (activeHashBits + 31) >> 5;
  clearBits(outBits, words);

  let winners = minI32(minI32(activeWinners, activeHashBits), MAX_WINNERS);
  for (let i: i32 = 0; i < winners; i++) {
    unchecked(topScores[i] = -f32.MAX_VALUE);
    unchecked(topIndices[i] = -1);
  }

  for (let kc: i32 = 0; kc < activeHashBits; kc++) {
    let score: f32 = 0.0;

    for (let p: i32 = 0; p < activeProjections; p++) {
      let mix = hash32(activeSeed ^ (<u32>kc * 0x9E3779B1) ^ (<u32>p * 0x85EBCA6B));
      let featureIdx = <i32>(mix & 0xFF);
      let sign: f32 = (mix & 0x100) == 0 ? 1.0 : -1.0;
      score += sign * unchecked(features[featureIdx]);
    }

    insertTop(score, kc, winners);
  }

  for (let i: i32 = 0; i < winners; i++) {
    let kc = unchecked(topIndices[i]);
    if (kc < 0) continue;

    let word = kc >> 5;
    let bit = kc & 31;
    unchecked(outBits[word] |= 1 << bit);
  }
}

function overlapBits(a: StaticArray<u32>, b: StaticArray<u32>): i32 {
  let words = (activeHashBits + 31) >> 5;
  let overlap: i32 = 0;

  // WTA similarity as Hamming intersection: number of active hash bits shared by both hashes.
  for (let i: i32 = 0; i < words; i++) {
    let bits = unchecked(a[i] & b[i]);
    overlap += popcnt<u32>(bits);
  }

  return overlap;
}

export function configure(hashBits: i32 = 512, winners: i32 = 48, projections: i32 = 6, seed: u32 = 0xC0FFEE): void {
  if (hashBits < 64) hashBits = 64;
  if (hashBits > MAX_HASH_BITS) hashBits = MAX_HASH_BITS;

  if (winners < 8) winners = 8;
  if (winners > MAX_WINNERS) winners = MAX_WINNERS;

  if (projections < 2) projections = 2;
  if (projections > MAX_PROJECTIONS) projections = MAX_PROJECTIONS;

  activeHashBits = hashBits;
  activeWinners = winners;
  activeProjections = projections;
  activeSeed = seed;
}

export function queryBufferPtr(): usize {
  return queryPtr;
}

export function candidateBufferPtr(): usize {
  return candidatePtr;
}

export function bufferCapacity(): i32 {
  return MAX_INPUT_BYTES;
}

export function scoreFromBuffers(queryLength: i32, candidateLength: i32): i32 {
  let qLen = minI32(queryLength, MAX_INPUT_BYTES);
  if (qLen < 0) qLen = 0;

  let cLen = minI32(candidateLength, MAX_INPUT_BYTES);
  if (cLen < 0) cLen = 0;
  if (qLen == 0 || cLen == 0) return 0;

  buildFeatures(queryPtr, qLen, queryFeatures);
  buildFeatures(candidatePtr, cLen, candidateFeatures);

  hashFeatures(queryFeatures, queryBits);
  hashFeatures(candidateFeatures, candidateBits);

  return overlapBits(queryBits, candidateBits);
}
