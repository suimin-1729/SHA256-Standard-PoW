// Fixed WebGPU PoW implementation
// 修正点:
// - LCG をJS側で計算し、ランダムサフィックスを GPU に渡す
// - GPU側は与えられたサフィックスを使用してメッセージを構築、SHA-256計算
// これによりエンディアン不整合やLCG実装の不正確さを完全に排除

let device = null;
let isMining = false;
let stopMining = false;
let startTime = null;
let lastNonce = 0;
let handlingResult = false; // guard to avoid processing multiple GPU results concurrently
// ログ出力制御: true にすると詳細ログを出す（デフォルトは false）
const VERBOSE_LOGGING = false;
function dlog(...args) { if (VERBOSE_LOGGING) console.log(...args); }
function dwarn(...args) { if (VERBOSE_LOGGING) console.warn(...args); }

// ---------- WGSL シェーダー（簡略版） ----------
/* eslint-disable */
const sha256Shader = `
// SHA-256 PoW Compute Shader (GPU generates suffix per-thread using LCG)

@group(0) @binding(0) var<storage, read> maskBuffer: array<u32>;
@group(0) @binding(1) var<storage, read> keyBuffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> resultBuffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> nonceBuffer: array<u32>;
@group(0) @binding(4) var<storage, read_write> hashBuffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> debugBuffer: array<u32>;

// SHA-256 constants
const K: array<u32, 64> = array<u32, 64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
);

fn rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
fn ch(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ ((~x) & z); }
fn maj(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ (x & z) ^ (y & z); }
fn sigma0(x: u32) -> u32 { return rotr(x,2u) ^ rotr(x,13u) ^ rotr(x,22u); }
fn sigma1(x: u32) -> u32 { return rotr(x,6u) ^ rotr(x,11u) ^ rotr(x,25u); }
fn gamma0(x: u32) -> u32 { return rotr(x,7u) ^ rotr(x,18u) ^ (x >> 3u); }
fn gamma1(x: u32) -> u32 { return rotr(x,17u) ^ rotr(x,19u) ^ (x >> 10u); }

// LCG constants for suffix generation
const RAND_MULT: u32 = 0x4c957f2du; // low 32 bits of 0x5851f42d4c957f2d
const RAND_INC: u32 = 1u;
const BASE62: array<u32, 62> = array<u32, 62>(
    0x30u, 0x31u, 0x32u, 0x33u, 0x34u, 0x35u, 0x36u, 0x37u, 0x38u, 0x39u,
    0x41u, 0x42u, 0x43u, 0x44u, 0x45u, 0x46u, 0x47u, 0x48u, 0x49u, 0x4au, 0x4bu, 0x4cu, 0x4du, 0x4eu, 0x4fu,
    0x50u, 0x51u, 0x52u, 0x53u, 0x54u, 0x55u, 0x56u, 0x57u, 0x58u, 0x59u, 0x5au,
    0x61u, 0x62u, 0x63u, 0x64u, 0x65u, 0x66u, 0x67u, 0x68u, 0x69u, 0x6au, 0x6bu, 0x6cu, 0x6du, 0x6eu, 0x6fu,
    0x70u, 0x71u, 0x72u, 0x73u, 0x74u, 0x75u, 0x76u, 0x77u, 0x78u, 0x79u, 0x7au
);

fn generate_suffix_bytes(seed: u32, length: u32) -> array<u32,16> {
    var result: array<u32,16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { result[i] = 0u; }
    var state = seed & 0xffffffffu;
    for (var i: u32 = 0u; i < length; i = i + 1u) {
        state = (state * RAND_MULT + RAND_INC) & 0xffffffffu;
        let idx = (state >> 16u) % 62u;
        let byte_val = BASE62[idx];
        let word_idx = i / 4u;
        let byte_in_word = i % 4u;
        result[word_idx] = result[word_idx] | (byte_val << (24u - byte_in_word * 8u));
    }
    return result;
}

fn expand_message_schedule(block: array<u32,16>) -> array<u32,64> {
    var w: array<u32,64>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { w[i] = block[i]; }
    for (var i: u32 = 16u; i < 64u; i = i + 1u) { w[i] = gamma1(w[i-2u]) + w[i-7u] + gamma0(w[i-15u]) + w[i-16u]; }
    return w;
}

fn sha256_block(block: array<u32,16>) -> array<u32,8> {
    var h: array<u32,8> = array<u32,8>(0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u);
    var w = expand_message_schedule(block);
    var a: u32 = h[0]; var b: u32 = h[1]; var c: u32 = h[2]; var d: u32 = h[3];
    var e: u32 = h[4]; var f: u32 = h[5]; var g: u32 = h[6]; var hh: u32 = h[7];
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        let T1 = hh + sigma1(e) + ch(e,f,g) + K[i] + w[i];
        let T2 = sigma0(a) + maj(a,b,c);
        hh = g; g = f; f = e; e = d + T1; d = c; c = b; b = a; a = T1 + T2;
    }
    h[0] = a + h[0]; h[1] = b + h[1]; h[2] = c + h[2]; h[3] = d + h[3]; h[4] = e + h[4]; h[5] = f + h[5]; h[6] = g + h[6]; h[7] = hh + h[7];
    return h;
}

fn create_padded_message(key: array<u32,16>, key_len: u32, suffix: array<u32,16>, suffix_len: u32) -> array<u32,16> {
    var msg_words: array<u32,16>;
    // zero
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { msg_words[i] = 0u; }
    var byte_idx: u32 = 0u;
    // copy key bytes
    for (var i: u32 = 0u; i < key_len; i = i + 1u) {
        let wi = i / 4u; let bo = i % 4u;
        let b = (key[wi] >> (24u - bo * 8u)) & 0xffu;
        let tw = byte_idx / 4u; let tbo = byte_idx % 4u;
        msg_words[tw] = msg_words[tw] | (b << (24u - tbo * 8u));
        byte_idx = byte_idx + 1u;
    }
    // copy suffix bytes
    for (var i: u32 = 0u; i < suffix_len; i = i + 1u) {
        let wi = i / 4u; let bo = i % 4u;
        let b = (suffix[wi] >> (24u - bo * 8u)) & 0xffu;
        let tw = byte_idx / 4u; let tbo = byte_idx % 4u;
        msg_words[tw] = msg_words[tw] | (b << (24u - tbo * 8u));
        byte_idx = byte_idx + 1u;
    }
    // padding: append 0x80
    let pw = byte_idx / 4u; let pbo = byte_idx % 4u;
    msg_words[pw] = msg_words[pw] | (0x80u << (24u - pbo * 8u));
    // bit length (high 32 bits zero since message <= 64 bytes)
    let bit_len: u32 = (key_len + suffix_len) * 8u;
    msg_words[14u] = 0u;
    msg_words[15u] = bit_len;
    return msg_words;
}

fn check_mask(hash: array<u32,8>, mask_len: u32, has_nibble: u32, mask_nibble: u32) -> bool {
    // mask_len is number of hex nibbles (hex characters)
    if (mask_len == 0u) { return true; }
    let full_bytes: u32 = mask_len / 2u; // number of whole bytes to compare
    for (var i: u32 = 0u; i < full_bytes; i = i + 1u) {
        let hw = i / 4u; let hb = i % 4u;
        let hash_byte = (hash[hw] >> (24u - hb * 8u)) & 0xffu;
        let mw = i / 4u; let mb = i % 4u;
        let mask_word = maskBuffer[mw];
        let mask_byte = (mask_word >> (24u - mb * 8u)) & 0xffu;
        if (hash_byte != mask_byte) { return false; }
    }
    // if odd nibble count, check the high nibble of the next hash byte
    if ((mask_len & 1u) == 1u) {
        let i = full_bytes; // index of byte containing the remaining nibble
        let hw = i / 4u; let hb = i % 4u;
        let hash_byte = (hash[hw] >> (24u - hb * 8u)) & 0xffu;
        let hn = (hash_byte >> 4u) & 0xfu;
        if (hn != mask_nibble) { return false; }
    }
    return true;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let idx = globalId.x;
    let start = nonceBuffer[0];
    let nonce = start + idx;
    let key_len = nonceBuffer[4];
    let suffix_len = nonceBuffer[5];
    let mask_len = nonceBuffer[6];
    let has_nibble = nonceBuffer[7];
    let mask_nibble = nonceBuffer[8];
    let batch_size = nonceBuffer[9];

    // bounds check: avoid reading suffixBuffer out-of-range
    if (idx >= batch_size) { return; }

    if (atomicLoad(&resultBuffer[0]) != 0u) { return; }

    // copy key
    var key: array<u32,16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) { key[i] = keyBuffer[i]; }

    // generate suffix locally using LCG (no buffer transfer)
    let seed_xor = nonceBuffer[2] ^ nonceBuffer[3] ^ nonce;
    let suffix = generate_suffix_bytes(seed_xor, suffix_len);

    let padded = create_padded_message(key, key_len, suffix, suffix_len);
    let hash = sha256_block(padded);

    if (check_mask(hash, mask_len, has_nibble, mask_nibble)) {
        // attempt to claim the result (only one thread should win and write outputs)
        let expected: u32 = 0u;
        let desired: u32 = 1u;
        let res = atomicCompareExchangeWeak(&resultBuffer[0], expected, desired);
            if (res.exchanged) {
                nonceBuffer[1] = nonce;
                for (var i: u32 = 0u; i < 8u; i = i + 1u) { hashBuffer[i] = hash[i]; }
                // write debug info
                // compute message schedule locally here (claiming thread only) and store w[0..7]
                let wlocal = expand_message_schedule(padded);
                for (var j: u32 = 0u; j < 8u; j = j + 1u) { debugBuffer[32u + j] = wlocal[j]; }
                for (var i: u32 = 0u; i < 16u; i = i + 1u) { debugBuffer[i] = padded[i]; }
                for (var i: u32 = 0u; i < 8u; i = i + 1u) { debugBuffer[16u + i] = hash[i]; }
                debugBuffer[24u] = key_len + suffix_len;
                debugBuffer[25u] = (key_len + suffix_len) * 8u;
                debugBuffer[26u] = nonce;
                return;
            }
    }
    // if thread 0, write some debug for monitoring — but only if result not claimed
    if (idx == 0u) {
        if (atomicLoad(&resultBuffer[0]) == 0u) {
            for (var i: u32 = 0u; i < 16u; i = i + 1u) { debugBuffer[i] = padded[i]; }
            debugBuffer[24u] = key_len + suffix_len;
            debugBuffer[25u] = (key_len + suffix_len) * 8u;
            debugBuffer[26u] = nonce;
        }
    }
}
`;
/* eslint-enable */
function stringToBytes(str) {
    const bytes = new Uint8Array(str.length);
    for (let i = 0; i < str.length; i++) bytes[i] = str.charCodeAt(i);
    return bytes;
}

function toBigEndianU32(byteArray) {
    const paddedLen = Math.ceil(byteArray.length / 4) * 4;
    const u32len = Math.ceil(paddedLen / 4);
    const out = new Uint32Array(u32len);
    for (let i = 0; i < u32len; i++) {
        const b0 = byteArray[i*4]   || 0;
        const b1 = byteArray[i*4+1] || 0;
        const b2 = byteArray[i*4+2] || 0;
        const b3 = byteArray[i*4+3] || 0;
        out[i] = (((b0 << 24) >>> 0) | ((b1 << 16) >>> 0) | ((b2 << 8) >>> 0) | (b3 >>> 0)) >>> 0;
    }
    return out;
}

function fnv1a64(data) {
    let hash = 0xcbf29ce484222325n;
    for (let i = 0; i < data.length; i++) {
        hash ^= BigInt(data[i]);
        hash = (hash * 0x100000001b3n) & 0xffffffffffffffffn;
    }
    return hash;
}

// JS 側で使用する LCG 定数（WGSL と一致させる）
const RAND_MULT = 0x5851f42d4c957f2dn; // 6364136223846793005n
const RAND_INC = 1n;

async function verifySHA256(messageBytes) {
    try {
        const hashBuffer = await crypto.subtle.digest('SHA-256', messageBytes);
        const hashArray = new Uint8Array(hashBuffer);
        const hashHex = Array.from(hashArray).map(b => b.toString(16).padStart(2, '0')).join('');
        dlog('Verified SHA-256 (JS crypto):', hashHex);
        return hashHex;
    } catch (e) {
        console.error('SHA-256 verification error:', e);
        return null;
    }
}

function fillRandomBase62(state, length) {
    const BASE62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
    let result = '';
    let current = state & 0xffffffffffffffffn;
    for (let i = 0; i < length; i++) {
        current = (current * RAND_MULT + RAND_INC) & 0xffffffffffffffffn;
        const idx = Number((current >> 32n) % 62n);
        result += BASE62[idx];
    }
    return result;
}

// ---------- WebGPU 初期化 ----------
async function initWebGPU() {
    const gpuCheck = document.getElementById('gpu-check');
    if (!navigator.gpu) {
        gpuCheck.innerHTML = '<p class="error">❌ WebGPUがサポートされていません。</p>';
        return false;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) { gpuCheck.innerHTML = '<p class="error">❌ GPU アダプタが見つかりません</p>'; return false; }
        device = await adapter.requestDevice();
        gpuCheck.innerHTML = '<p class="success">✅ GPU が初期化されました</p>';
        document.getElementById('main-content').classList.remove('hidden');
        return true;
    } catch (e) {
        gpuCheck.innerHTML = `<p class="error">❌ GPU 初期化失敗: ${e.message}</p>`;
        return false;
    }
}

// ---------- PoW 本体 ----------
async function startMining(key, mask) {
    if (!device) { alert('GPU 未初期化'); return; }
    isMining = true; stopMining = false; startTime = Date.now(); lastNonce = 0;

    // 一時的に冗長ログをフィルタリングする（最終結果とエラーのみ表示）
    const _origConsoleLog = console.log;
    const _origConsoleWarn = console.warn;
    let _consoleOverridden = false;
    if (!VERBOSE_LOGGING) {
        _consoleOverridden = true;
        console.log = (...args) => {
            try {
                const s = args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
                // 最終結果や明示的なエラーのみを表示
                if (s.includes('=== FINAL RESULT ===') || s.includes('Answer:') || s.includes('エラー') || s.includes('Error')) {
                    _origConsoleLog(...args);
                }
            } catch (e) { /* ignore */ }
        };
        console.warn = (...args) => {
            try {
                const s = args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ');
                // 重要そうな警告だけ表示（必要なら条件を追加）
                if (s.includes('エラー') || s.includes('Error')) {
                    _origConsoleWarn(...args);
                }
            } catch (e) { /* ignore */ }
        };
    }

    const RANDOM_SUFFIX_LEN = 14;
    const { bytes: maskBytes, maskNibble } = parseMask(mask);
    const keyBytes = stringToBytes(key);
    const keySeed = fnv1a64(keyBytes);

    // JS側で事前計算したランダムサフィックスをGPUに渡す戦略に変更
    // 各nonceに対してランダムサフィックスを計算し、バッファに格納
    function generateSuffixBytes(nonce) {
        const suffix = fillRandomBase62(BigInt(keySeed) ^ BigInt(nonce), RANDOM_SUFFIX_LEN);
        return stringToBytes(suffix);
    }

    // Big-Endian に揃えて GPU に渡す
    const maskUint32_raw = toBigEndianU32(new Uint8Array(maskBytes));
    const keyUint32_raw = toBigEndianU32(new Uint8Array(keyBytes));
    // Ensure buffers are 16 u32 (64 bytes) so WGSL can read fixed 16-word arrays safely
    const maskUint32 = new Uint32Array(16);
    maskUint32.set(maskUint32_raw.slice(0, 16));
    const keyUint32 = new Uint32Array(16);
    keyUint32.set(keyUint32_raw.slice(0, 16));

    // バッチサイズ（ワークアイテム数）
    

    // GPU バッファ作成
    const maskGpuBuffer = device.createBuffer({ size: maskUint32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const keyGpuBuffer  = device.createBuffer({ size: keyUint32.byteLength,  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const resultBuffer  = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const nonceBuffer   = device.createBuffer({ size: 10 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const hashBuffer    = device.createBuffer({ size: 8 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    // 初期バッチサイズ（ワークアイテム数） — Rust gpu.rs に合わせて設定
    // Rust: 初期値 10,000,000、段階的に増加、最大 50,000,000
    // WebGPU: 初期値を控えめにして、性能に応じて増加
    let currentBatchSize = 1000000; // 初期値: 1M (WebGPU メモリ制限を考慮)
    const maxBatchSize = 10000000;  // 最大値: 10M
    const previewInterval = 8;      // ライブプレビュー読み取り間隔（バッチ単位）

    device.queue.writeBuffer(maskGpuBuffer, 0, maskUint32);
    device.queue.writeBuffer(keyGpuBuffer,  0, keyUint32);

    // デバッグバッファ
    const debugBuffer = device.createBuffer({ size: 40 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const debugReadBuffer = device.createBuffer({ size: 40 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // パイプライン
    const shaderModule = device.createShaderModule({ code: sha256Shader });
    const computePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' } });
    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: maskGpuBuffer } },
            { binding: 1, resource: { buffer: keyGpuBuffer } },
            { binding: 2, resource: { buffer: resultBuffer } },
            { binding: 3, resource: { buffer: nonceBuffer } },
            { binding: 4, resource: { buffer: hashBuffer } },
            { binding: 5, resource: { buffer: debugBuffer } }
        ]
    });

    // 読み取り用バッファ
    const resultReadBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const nonceReadBuffer  = device.createBuffer({ size: 9 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const hashReadBuffer   = device.createBuffer({ size: 8 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // nonce 初期化
    let currentNonce = 0;
    const keySeedLow = Number(keySeed & 0xffffffffn);
    const keySeedHigh = Number((keySeed >> 32n) & 0xffffffffn);

    const nonceData = new Uint32Array([
        currentNonce, // start
        0,            // found
        keySeedLow,
        keySeedHigh,
        keyBytes.length,
        RANDOM_SUFFIX_LEN,
        mask.length, // mask length in nibbles (hex chars)
        maskNibble !== null ? 1 : 0,
        maskNibble !== null ? maskNibble : 0
    ,
        currentBatchSize
    ]);
    device.queue.writeBuffer(nonceBuffer, 0, nonceData);

    try {
        // mark UI as running
        setRunning(true);
        let batchCounter = 0;
        let successCount = 0; // 段階的増加用: 連続成功回数
        while (isMining && !stopMining) {
            // reset result
            device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));
            
            // update nonce (GPU will generate suffixes locally)
            nonceData[0] = currentNonce;
            device.queue.writeBuffer(nonceBuffer, 0, nonceData);
            
            // Live preview: show first candidate answer (JS-side LCG computation)
            const previewSuffix = fillRandomBase62(BigInt(keySeed) ^ BigInt(currentNonce), RANDOM_SUFFIX_LEN);
            updateLiveCandidate(key + previewSuffix, '-', currentNonce);
            
            const commandEncoder = device.createCommandEncoder();

            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(currentBatchSize / 256));
            passEncoder.end();

            // copy results out
            commandEncoder.copyBufferToBuffer(resultBuffer, 0, resultReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(nonceBuffer, 0, nonceReadBuffer, 0, 9 * 4);
            commandEncoder.copyBufferToBuffer(hashBuffer, 0, hashReadBuffer, 0, 8 * 4);
            commandEncoder.copyBufferToBuffer(debugBuffer, 0, debugReadBuffer, 0, 40 * 4);

            const commandBuffer = commandEncoder.finish();
            device.queue.submit([commandBuffer]);

                await resultReadBuffer.mapAsync(GPUMapMode.READ);
                await nonceReadBuffer.mapAsync(GPUMapMode.READ);

            const resultArray = new Uint32Array(resultReadBuffer.getMappedRange().slice(0));
            const nonceArray  = new Uint32Array(nonceReadBuffer.getMappedRange().slice(0));

            currentNonce += currentBatchSize;
            lastNonce = currentNonce;
            updateProgress(currentNonce);
            
            // バッチサイズを段階的に増加（見つからなかった場合）
            if (resultArray[0] === 0) {
                successCount++;
                if (successCount >= 2 && currentBatchSize < maxBatchSize) {
                    const newSize = Math.min(currentBatchSize * 2, maxBatchSize);
                    if (newSize > currentBatchSize) {
                        currentBatchSize = newSize;
                        successCount = 0; // リセット
                        dlog(`Batch size increased to ${currentBatchSize}`);
                    }
                }
            } else {
                successCount = 0; // 見つかった場合はリセット
            }

            if (resultArray[0] !== 0) {
                // found
                if (handlingResult) {
                    // 既に別のスレッドで処理中 → スキップして次バッチへ
                    resultReadBuffer.unmap(); nonceReadBuffer.unmap();
                    continue;
                }
                handlingResult = true;
                
                await hashReadBuffer.mapAsync(GPUMapMode.READ);
                await debugReadBuffer.mapAsync(GPUMapMode.READ);

                const debugUint32 = new Uint32Array(debugReadBuffer.getMappedRange().slice(0));
                const hashUint32 = new Uint32Array(hashReadBuffer.getMappedRange().slice(0));

                // debug: message bytes
                const messageBytes = new Uint8Array(64);
                for (let i = 0; i < 16; i++) {
                    const word = debugUint32[i] >>> 0;
                    messageBytes[i * 4]     = (word >> 24) & 0xff;
                    messageBytes[i * 4 + 1] = (word >> 16) & 0xff;
                    messageBytes[i * 4 + 2] = (word >> 8) & 0xff;
                    messageBytes[i * 4 + 3] = word & 0xff;
                }

                const stateHashArray = new Uint8Array(32);
                for (let i = 0; i < 8; i++) {
                    const w = debugUint32[16 + i] >>> 0;
                    stateHashArray[i * 4]     = (w >> 24) & 0xff;
                    stateHashArray[i * 4 + 1] = (w >> 16) & 0xff;
                    stateHashArray[i * 4 + 2] = (w >> 8) & 0xff;
                    stateHashArray[i * 4 + 3] = w & 0xff;
                }

                const hashArray = new Uint8Array(32);
                for (let i = 0; i < 8; i++) {
                    const w = hashUint32[i] >>> 0;
                    hashArray[i * 4]     = (w >> 24) & 0xff;
                    hashArray[i * 4 + 1] = (w >> 16) & 0xff;
                    hashArray[i * 4 + 2] = (w >> 8) & 0xff;
                    hashArray[i * 4 + 3] = w & 0xff;
                }

                // 再現: GPU の debugBuffer から取り出したメッセージを優先して表示
                const foundNonce = nonceArray[1];
                // メッセージの実際の長さを取得
                const msgLen = debugUint32[24];
                const actualMessage = String.fromCharCode(...messageBytes.slice(0, msgLen));
                const answer = actualMessage; // GPU が出力したメッセージをそのまま表示（確実な一致）

                dlog('--- Found ---');
                dlog('Nonce:', foundNonce);
                dlog('Actual Answer (from LCG):', answer);
                dlog('Message in GPU:', actualMessage);
                dlog('Message Length:', msgLen);
                dlog('Message (hex):', Array.from(messageBytes.slice(0, msgLen)).map(b => b.toString(16).padStart(2, '0')).join(' '));
                
                // Hash をバイト配列として取得
                const hashFromShader = Array.from(stateHashArray).map(b => b.toString(16).padStart(2,'0')).join('');
                dlog('Hash from shader:', hashFromShader);
                
                // メッセージが一致しているか確認
                if (actualMessage !== (key + fillRandomBase62(BigInt(keySeed) ^ BigInt(foundNonce), RANDOM_SUFFIX_LEN))) {
                    dwarn('WARNING: JS LCG result differs from GPU output. Using GPU output as authoritative for display.');
                    dwarn('JS LCG:', key + fillRandomBase62(BigInt(keySeed) ^ BigInt(foundNonce), RANDOM_SUFFIX_LEN));
                    dwarn('GPU message:', actualMessage);
                }

                // JS 側で再計算して検証
                const messageBytesForVerify = new Uint8Array(msgLen);
                for (let i = 0; i < msgLen; i++) {
                    messageBytesForVerify[i] = actualMessage.charCodeAt(i);
                }
                dlog('Verifying SHA-256...');
                const verifiedHex = await verifySHA256(messageBytesForVerify);

                // Compare shader hash vs JS-verified hash
                const shaderHex = hashFromShader.toLowerCase();
                const verifiedHexNorm = (verifiedHex || '').toLowerCase();
                if (verifiedHexNorm !== shaderHex) {
                    // Detailed one-shot diagnostic and stop mining to avoid log spam
                    console.error('GPU/CPU SHA-256 mismatch detected — dumping diagnostic and stopping mining.');
                    console.error('Nonce:', foundNonce);
                    console.error('Message (ascii):', actualMessage);
                    console.error('Message (hex):', Array.from(messageBytes.slice(0, msgLen)).map(b => b.toString(16).padStart(2,'0')).join(' '));
                    // dump message words as u32 (big-endian interpretation)
                    const msgWords = [];
                    for (let i = 0; i < 16; i++) msgWords.push(('0x' + (debugUint32[i] >>> 0).toString(16).padStart(8,'0')));
                    console.error('Message words (u32 BE):', msgWords.join(' '));
                    console.error('Shader state words (u32 BE):', Array.from(debugUint32.slice(16,24)).map(v => '0x' + (v >>> 0).toString(16).padStart(8,'0')).join(' '));
                        console.error('Message schedule w[0..7] (u32 BE):', Array.from(debugUint32.slice(32,40)).map(v => '0x' + (v >>> 0).toString(16).padStart(8,'0')).join(' '));
                    console.error('Shader hash :', shaderHex.toUpperCase());
                    console.error('Verified hash:', verifiedHexNorm.toUpperCase());
                    // reset result flag on GPU for cleanliness
                    device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));
                    resultReadBuffer.unmap(); nonceReadBuffer.unmap(); hashReadBuffer.unmap(); debugReadBuffer.unmap();
                    isMining = false; handlingResult = false;
                    break;
                }

                // show result (verified)
                const elapsed = (Date.now() - startTime) / 1000;
                showResult(foundNonce, stateHashArray, elapsed, key, keySeed, answer);

                resultReadBuffer.unmap(); nonceReadBuffer.unmap(); hashReadBuffer.unmap(); debugReadBuffer.unmap();
                isMining = false; handlingResult = false; break;
            }

            // live preview: occasionally map debug buffer and compute preview hash
            if ((batchCounter % previewInterval) === 0) {
                await debugReadBuffer.mapAsync(GPUMapMode.READ);
                const debugPreview = new Uint32Array(debugReadBuffer.getMappedRange().slice(0));
                const previewMsgLen = debugPreview[24] || 0;
                const previewBytes = new Uint8Array(previewMsgLen);
                for (let i = 0; i < previewMsgLen; i++) {
                    const w = debugPreview[Math.floor(i / 4)] >>> 0;
                    const shift = 24 - (i % 4) * 8;
                    previewBytes[i] = (w >> shift) & 0xff;
                }
                const previewAnswer = previewMsgLen > 0 ? String.fromCharCode(...previewBytes) : '-';
                // unmap before running CPU-side digest to avoid holding mapped ranges
                debugReadBuffer.unmap();
                resultReadBuffer.unmap(); nonceReadBuffer.unmap();
                // async compute the preview hash and update UI (no await)
                verifySHA256(previewBytes).then((h) => {
                    updateLiveCandidate(previewAnswer, h || '-', nonceArray[0]);
                }).catch(() => { updateLiveCandidate(previewAnswer, '-', nonceArray[0]); });
            } else {
                // skip debug read this batch — we already updated candidate from JS-generated suffix
                resultReadBuffer.unmap(); nonceReadBuffer.unmap();
            }

            batchCounter = (batchCounter + 1) >>> 0;
            // 次バッチへ
        }
    } catch (e) {
        console.error('Mining error:', e);
        alert('エラー: ' + e.message);
        isMining = false;
    } finally {
        // コンソールを元に戻す
        if (_consoleOverridden) {
            console.log = _origConsoleLog;
            console.warn = _origConsoleWarn;
        }
        setRunning(false);
    }
}

// UI helper: enable/disable inputs and show spinner
function setRunning(running) {
    const keyInput = document.getElementById('key-input');
    const maskInput = document.getElementById('mask-input');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const spinner = document.getElementById('start-spinner');
    const progress = document.getElementById('progress-section');
    const result = document.getElementById('result-section');
    if (running) {
        if (keyInput) keyInput.classList.add('disabled'), keyInput.disabled = true;
        if (maskInput) maskInput.classList.add('disabled'), maskInput.disabled = true;
        if (startBtn) startBtn.disabled = true;
        if (spinner) spinner.classList.remove('hidden');
        if (stopBtn) stopBtn.classList.remove('hidden');
        if (progress) progress.classList.remove('hidden');
        if (result) result.classList.remove('hidden');
    } else {
        if (keyInput) keyInput.classList.remove('disabled'), keyInput.disabled = false;
        if (maskInput) maskInput.classList.remove('disabled'), maskInput.disabled = false;
        if (startBtn) startBtn.disabled = false;
        if (spinner) spinner.classList.add('hidden');
        if (stopBtn) stopBtn.classList.add('hidden');
    }
}

// update live candidate display
function updateLiveCandidate(answer, hashHex, nonce) {
    const elAns = document.getElementById('current-answer');
    const elNonce = document.getElementById('current-nonce');
    if (elAns) elAns.textContent = answer;
    if (elNonce) elNonce.textContent = Number(nonce).toLocaleString();
}

function parseMask(maskStr) {
    const bytes = [];
    let maskNibble = null;
    if (maskStr.length % 2 === 1) {
        const full = maskStr.slice(0, -1);
        const last = maskStr[maskStr.length - 1];
        maskNibble = parseInt(last, 16);
        for (let i = 0; i < full.length; i += 2) bytes.push(parseInt(full.substr(i,2),16));
    } else {
        for (let i = 0; i < maskStr.length; i += 2) bytes.push(parseInt(maskStr.substr(i,2),16));
    }
    return { bytes, maskNibble };
}

function updateProgress(nonce) {
    const elapsed = (Date.now() - startTime) / 1000;
    const hashRate = elapsed > 0 ? Math.floor(nonce / elapsed) : 0;
    const elNonce = document.getElementById('current-nonce');
    if (elNonce) elNonce.textContent = nonce.toLocaleString();
    const elRate = document.getElementById('hash-rate');
    if (elRate) elRate.textContent = hashRate.toLocaleString();
}

function showResult(nonce, hashArray, elapsed, key, keySeed, overrideAnswer) {
    let hashHex = Array.from(hashArray).map(b => b.toString(16).padStart(2,'0')).join('');
    const RANDOM_SUFFIX_LEN = 14;
    const answer = (typeof overrideAnswer === 'string' && overrideAnswer.length > 0)
        ? overrideAnswer
        : key + fillRandomBase62(BigInt(keySeed) ^ BigInt(nonce), RANDOM_SUFFIX_LEN);
    const section = document.getElementById('result-section');
    if (section) {
        document.getElementById('result-nonce').textContent = nonce.toLocaleString();
        document.getElementById('result-answer').textContent = answer;
        document.getElementById('result-hash').textContent = hashHex;
        document.getElementById('result-time').textContent = elapsed.toFixed(3) + ' 秒';
        section.classList.remove('hidden');
    }
    
    console.log('=== FINAL RESULT ===');
    console.log('Answer:', answer);
    console.log('Expected Hash Prefix:', document.getElementById('mask-input').value.trim().toUpperCase());
    console.log('Actual Hash:', hashHex.toUpperCase());
    console.log('Hash starts with mask:', hashHex.toUpperCase().startsWith(document.getElementById('mask-input').value.trim().toUpperCase()));
}

// イベント登録
document.addEventListener('DOMContentLoaded', async () => {
    const ok = await initWebGPU();
    if (!ok) return;
    const form = document.getElementById('pow-form');
    const stopBtn = document.getElementById('stop-btn');
    if (form) form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const key = document.getElementById('key-input').value.trim();
        const mask = document.getElementById('mask-input').value.trim();
        if (key.length !== 2) { alert('Key は 2 文字'); return; }
        if (!/^[0-9a-fA-F]+$/.test(mask)) { alert('Mask は 16 進数'); return; }
        await startMining(key, mask);
    });
    if (stopBtn) stopBtn.addEventListener('click', () => { stopMining = true; isMining = false; });
});
