// Fixed WebGPU PoW implementation (pow_fixed.js)
// 修正点まとめ:
// - JS と WGSL 間のエンディアン不整合を解消（key/mask を Big-Endian 表示で GPU に渡す）
// - LCG の乗数を JS / WGSL 両方で一致させた (0x5851f42d4c957f2d)
// - hashBuffer の usage 修正（STORAGE を含める）
// - デバッグ出力は残しました（問題追跡に便利なため）

let device = null;
let isMining = false;
let stopMining = false;
let startTime = null;
let lastNonce = 0;

// ---------- WGSL シェーダー（LCG 定数を修正、その他は大筋同等） ----------
const sha256Shader = `
@group(0) @binding(0) var<storage, read> maskBuffer: array<u32>;
@group(0) @binding(1) var<storage, read> keyBuffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> resultBuffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> nonceBuffer: array<u32>;
@group(0) @binding(4) var<storage, read_write> hashBuffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> debugBuffer: array<u32>;

// LCG 定数 (64-bit): 0x5851f42d4c957f2d
const RAND_MULT_LOW: u32 = 0x4c957f2du; // lower 32 bits
const RAND_MULT_HIGH: u32 = 0x5851f42du; // upper 32 bits
const RAND_INC_LOW: u32 = 1u;
const RAND_INC_HIGH: u32 = 0u;

fn mul_u64(a_low: u32, a_high: u32, b_low: u32, b_high: u32) -> vec2<u32> {
    // 32x32 => 64 部分積を組み合わせる単純実装（以前の実装を踏襲）
    let a_low_lo = a_low & 0xffffu;
    let a_low_hi = a_low >> 16u;
    let b_low_lo = b_low & 0xffffu;
    let b_low_hi = b_low >> 16u;

    let p00 = a_low_lo * b_low_lo;
    let p01 = a_low_lo * b_low_hi;
    let p10 = a_low_hi * b_low_lo;
    let p11 = a_low_hi * b_low_hi;

    let p0_low_part = p00 & 0xffffu;
    let p0_carry1 = p00 >> 16u;
    let p0_mid = p0_carry1 + (p01 & 0xffffu) + (p10 & 0xffffu);
    let p0_carry2 = p0_mid >> 16u;
    let p0_low = p0_low_part + ((p0_mid & 0xffffu) << 16u);
    let p0_high_part = p0_carry2 + (p01 >> 16u) + (p10 >> 16u) + p11;
    let p0_high = p0_high_part + select(0u, 1u, p0_low < p0_low_part);

    // a_low * b_high
    let p1_full_low = a_low * b_high;
    // a_high * b_low
    let p2_full_low = a_high * b_low;
    let p3_val = a_high * b_high;

    var result_low = p0_low;
    var result_high = p0_high;

    let sum1 = result_high + p1_full_low;
    let carry1 = select(0u, 1u, sum1 < result_high);
    result_high = sum1;

    let sum2 = result_high + p2_full_low;
    let carry2 = select(0u, 1u, sum2 < result_high);
    result_high = sum2 + carry1;
    let carry2_total = carry2 + select(0u, 1u, result_high < sum2);

    let sum3 = result_high + p3_val;
    let carry3 = select(0u, 1u, sum3 < result_high);
    result_high = sum3 + carry2_total;

    return vec2<u32>(result_low, result_high);
}

fn add_u64(a_low: u32, a_high: u32, b_low: u32, b_high: u32) -> vec2<u32> {
    let sum_low = a_low + b_low;
    let carry = select(0u, 1u, sum_low < a_low);
    let sum_high = a_high + b_high + carry;
    return vec2<u32>(sum_low, sum_high);
}

const BASE62: array<u32, 62> = array<u32, 62>(
    48u,49u,50u,51u,52u,53u,54u,55u,56u,57u,
    65u,66u,67u,68u,69u,70u,71u,72u,73u,74u,75u,76u,77u,78u,79u,80u,81u,82u,83u,84u,85u,86u,87u,88u,89u,90u,
    97u,98u,99u,100u,101u,102u,103u,104u,105u,106u,107u,108u,109u,110u,111u,112u,113u,114u,115u,116u,117u,118u,119u,120u,121u,122u
);

const K: array<u32, 64> = array<u32, 64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82e4u, 0xab1c5ed5u,
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

fn rightRotate(value: u32, amount: u32) -> u32 {
    return (value >> amount) | (value << (32u - amount));
}

fn sha256_transform(state: ptr<function, array<u32, 8>>, chunk: ptr<function, array<u32, 16>>) {
    var w: array<u32, 64>;
    for (var i = 0u; i < 16u; i++) {
        w[i] = (*chunk)[i];
    }
    for (var i = 16u; i < 64u; i++) {
        let s0 = rightRotate(w[i - 15u], 7u) ^ rightRotate(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
        let s1 = rightRotate(w[i - 2u], 17u) ^ rightRotate(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }

    var a = (*state)[0];
    var b = (*state)[1];
    var c = (*state)[2];
    var d = (*state)[3];
    var e = (*state)[4];
    var f = (*state)[5];
    var g = (*state)[6];
    var h_val = (*state)[7];

    for (var i = 0u; i < 64u; i++) {
        let S1 = rightRotate(e, 6u) ^ rightRotate(e, 11u) ^ rightRotate(e, 25u);
        let ch = (e & f) ^ (~e & g);
        let temp1 = h_val + S1 + ch + K[i] + w[i];
        let S0 = rightRotate(a, 2u) ^ rightRotate(a, 13u) ^ rightRotate(a, 22u);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = S0 + maj;

        h_val = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    (*state)[0] += a;
    (*state)[1] += b;
    (*state)[2] += c;
    (*state)[3] += d;
    (*state)[4] += e;
    (*state)[5] += f;
    (*state)[6] += g;
    (*state)[7] += h_val;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let index = globalId.x;
    let startNonce = nonceBuffer[0];
    let nonce = startNonce + index;
    let keySeedLow = nonceBuffer[2];
    let keySeedHigh = nonceBuffer[3];
    let keyLen = nonceBuffer[4];
    let randomLen = nonceBuffer[5];
    let maskLen = nonceBuffer[6];
    let hasNibble = nonceBuffer[7];
    let maskNibble = nonceBuffer[8];

    if (atomicLoad(&resultBuffer[0]) != 0u) {
        return;
    }

    var state: array<u32, 8> = array<u32, 8>(
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    );

    var messageBytes: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { messageBytes[i] = 0u; }

    // Key を Big-Endian のバイト配列として messageBytes にコピーする
    let keyWords = (keyLen + 3u) / 4u;
    for (var i = 0u; i < keyLen && i < 64u; i++) {
        let wordIdx = i / 4u;
        let byteIdx = i % 4u;
        if (wordIdx < keyWords) {
            let word = keyBuffer[wordIdx];
            // keyBuffer は JS 側で Big-Endian に詰めてある前提
            // messageBytes はビッグエンディアンで配置する
            let byteVal = (word >> ((3u - byteIdx) * 8u)) & 0xffu;
            let msgWordIdx = i / 4u;
            let msgByteIdx = i % 4u;
            messageBytes[msgWordIdx] = messageBytes[msgWordIdx] | (byteVal << ((3u - msgByteIdx) * 8u));
        }
    }

    // stateRand = keySeed ^ nonce
    var stateRandLow = keySeedLow ^ nonce;
    var stateRandHigh = keySeedHigh;

    for (var i = 0u; i < randomLen && (keyLen + i) < 64u; i++) {
        let mulResult = mul_u64(stateRandLow, stateRandHigh, RAND_MULT_LOW, RAND_MULT_HIGH);
        let addResult = add_u64(mulResult.x, mulResult.y, RAND_INC_LOW, RAND_INC_HIGH);
        stateRandLow = addResult.x;
        stateRandHigh = addResult.y;

        let idx = stateRandHigh % 62u;
        let byteVal = BASE62[idx] & 0xffu;
        let pos = keyLen + i;
        let msgWordIdx = pos / 4u;
        let msgByteIdx = pos % 4u;
        messageBytes[msgWordIdx] = messageBytes[msgWordIdx] | (byteVal << ((3u - msgByteIdx) * 8u));
    }

    let msgLen = keyLen + randomLen;

    // padding: 0x80
    if (msgLen < 64u) {
        let msgWordIdx = msgLen / 4u;
        let msgByteIdx = msgLen % 4u;
        messageBytes[msgWordIdx] = messageBytes[msgWordIdx] | (0x80u << ((3u - msgByteIdx) * 8u));
    }

    // bit length (64-bit big-endian)。ここでは上位 32bit は 0 として下位に bitLen を配置
    let bitLen = msgLen * 8u;
    if (msgLen < 56u) {
        // messageBytes[14] (bytes 56-59) 上位 32bit
        messageBytes[14] = 0u;
        // messageBytes[15] (bytes 60-63) 下位 32bit をビッグエンディアンとして格納
        // WGSL の messageBytes[] は "ビッグエンディアン表現のワード" を期待しているため
        // ここで下位 32bit をそのままセットしてよい
        messageBytes[15] = bitLen;
        sha256_transform(&state, &messageBytes);
    } else {
        sha256_transform(&state, &messageBytes);
        for (var i = 0u; i < 14u; i++) { messageBytes[i] = 0u; }
        messageBytes[14] = 0u;
        messageBytes[15] = bitLen;
        sha256_transform(&state, &messageBytes);
    }

    // ハッシュバイト抽出 & マスク比較
    var matches = true;
    let maskWords = (maskLen + 3u) / 4u;
    for (var i = 0u; i < maskLen && i < 32u; i++) {
        let wordIdx = i / 4u;
        let byteIdx = i % 4u;
        if (wordIdx < 8u) {
            let stateWord = state[wordIdx];
            let hashByte = (stateWord >> ((3u - byteIdx) * 8u)) & 0xffu;
            if (wordIdx < maskWords) {
                // maskBuffer は JS 側で Big-Endian 詰め込み済
                let maskWord = maskBuffer[wordIdx];
                let maskByte = (maskWord >> ((3u - byteIdx) * 8u)) & 0xffu;
                if (hashByte != maskByte) { matches = false; break; }
            } else { matches = false; break; }
        } else { matches = false; break; }
    }

    if (matches && hasNibble != 0u) {
        if (maskLen >= 32u) { matches = false; }
        else {
            let wordIdx = maskLen / 4u;
            let byteIdx = maskLen % 4u;
            if (wordIdx < 8u) {
                let stateWord = state[wordIdx];
                let upperNibble = (stateWord >> ((3u - byteIdx) * 8u + 4u)) & 0x0fu;
                if (upperNibble != maskNibble) { matches = false; }
            } else { matches = false; }
        }
    }

    if (matches) {
        var expected = 0u;
        var desired = 1u;
        var result = atomicCompareExchangeWeak(&resultBuffer[0], expected, desired);
        if (result.old_value == expected && result.exchanged) {
            nonceBuffer[1] = nonce;
            for (var i = 0u; i < 8u; i++) { hashBuffer[i] = state[i]; }
            // デバッグ情報
            for (var i = 0u; i < 16u; i++) { debugBuffer[i] = messageBytes[i]; }
            for (var i = 0u; i < 8u; i++) { debugBuffer[16 + i] = state[i]; }
            debugBuffer[24] = msgLen;
            debugBuffer[25] = bitLen;
            debugBuffer[26] = nonce;
            return;
        }
    }

    if (index == 0u) {
        for (var i = 0u; i < 16u; i++) { debugBuffer[i] = messageBytes[i]; }
        debugBuffer[24] = msgLen;
        debugBuffer[25] = bitLen;
        debugBuffer[26] = nonce;
        for (var i = 0u; i < 8u; i++) { debugBuffer[16 + i] = state[i]; }
    }
}
`;

// ---------- JS 側ユーティリティ ----------
function toBigEndianU32(bytes) {
    const padded = new Uint8Array(Math.ceil(bytes.length / 4) * 4);
    for (let i = 0; i < padded.length; i++) padded[i] = 0;
    for (let i = 0; i < bytes.length; i++) {
        const wordIdx = Math.floor(i / 4) * 4;
        const offset = i % 4;
        // GPU 側は messageWords を "ビッグエンディアンで保持する" 実装なので
        // 各ワード内のバイトは MSB -> LSB の順で配置する
        padded[wordIdx + (3 - offset)] = bytes[i];
    }
    return new Uint32Array(padded.buffer);
}

function stringToBytes(str) {
    const bytes = new Uint8Array(str.length);
    for (let i = 0; i < str.length; i++) bytes[i] = str.charCodeAt(i);
    return bytes;
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

    const RANDOM_SUFFIX_LEN = 14;
    const { bytes: maskBytes, maskNibble } = parseMask(mask);
    const keyBytes = stringToBytes(key);
    const keySeed = fnv1a64(keyBytes);

    // Big-Endian に揃えて GPU に渡す
    const maskUint32 = toBigEndianU32(new Uint8Array(maskBytes));
    const keyUint32 = toBigEndianU32(new Uint8Array(keyBytes));

    // GPU バッファ作成
    const maskGpuBuffer = device.createBuffer({ size: maskUint32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const keyGpuBuffer  = device.createBuffer({ size: keyUint32.byteLength,  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const resultBuffer  = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const nonceBuffer   = device.createBuffer({ size: 9 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const hashBuffer    = device.createBuffer({ size: 8 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

    device.queue.writeBuffer(maskGpuBuffer, 0, maskUint32.buffer ? maskUint32.buffer : maskUint32);
    device.queue.writeBuffer(keyGpuBuffer,  0, keyUint32.buffer  ? keyUint32.buffer  : keyUint32);

    // デバッグバッファ
    const debugBuffer = device.createBuffer({ size: 27 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const debugReadBuffer = device.createBuffer({ size: 27 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

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
        maskBytes.length,
        maskNibble !== null ? 1 : 0,
        maskNibble !== null ? maskNibble : 0
    ]);
    device.queue.writeBuffer(nonceBuffer, 0, nonceData);

    const batchSize = 256 * 256; // workitems

    try {
        while (isMining && !stopMining) {
            // reset result
            device.queue.writeBuffer(resultBuffer, 0, new Uint32Array([0]));
            // update start nonce
            nonceData[0] = currentNonce;
            device.queue.writeBuffer(nonceBuffer, 0, nonceData);

            const commandEncoder = device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(batchSize / 256));
            passEncoder.end();

            // copy results out
            commandEncoder.copyBufferToBuffer(resultBuffer, 0, resultReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(nonceBuffer, 0, nonceReadBuffer, 0, 9 * 4);
            commandEncoder.copyBufferToBuffer(hashBuffer, 0, hashReadBuffer, 0, 8 * 4);
            commandEncoder.copyBufferToBuffer(debugBuffer, 0, debugReadBuffer, 0, 27 * 4);

            const commandBuffer = commandEncoder.finish();
            device.queue.submit([commandBuffer]);

            await resultReadBuffer.mapAsync(GPUMapMode.READ);
            await nonceReadBuffer.mapAsync(GPUMapMode.READ);

            const resultArray = new Uint32Array(resultReadBuffer.getMappedRange().slice(0));
            const nonceArray  = new Uint32Array(nonceReadBuffer.getMappedRange().slice(0));

            currentNonce += batchSize;
            lastNonce = currentNonce;
            updateProgress(currentNonce);

            if (resultArray[0] !== 0) {
                // found
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

                // 再現: 実際のランダムサフィックスを JS 側で生成して表示
                const foundNonce = nonceArray[1];
                const actualStateRand = BigInt(keySeed) ^ BigInt(foundNonce);
                const randomSuffix = fillRandomBase62(actualStateRand, RANDOM_SUFFIX_LEN);
                const answer = key + randomSuffix;

                console.log('--- Found ---');
                console.log('Nonce:', foundNonce);
                console.log('Message (hex):', Array.from(messageBytes).map(b => b.toString(16).padStart(2, '0')).join(' '));
                console.log('Message as string:', String.fromCharCode(...messageBytes.slice(0, debugUint32[24])));
                console.log('Hash from shader state:', Array.from(stateHashArray).map(b => b.toString(16).padStart(2,'0')).join(''));
                console.log('Hash from hashBuffer:', Array.from(hashArray).map(b => b.toString(16).padStart(2,'0')).join(''));

                // show result
                const elapsed = (Date.now() - startTime) / 1000;
                showResult(foundNonce, hashArray, elapsed, key, keySeed);

                resultReadBuffer.unmap(); nonceReadBuffer.unmap(); hashReadBuffer.unmap(); debugReadBuffer.unmap();
                isMining = false; break;
            }

            resultReadBuffer.unmap(); nonceReadBuffer.unmap();
            // 次バッチへ
        }
    } catch (e) {
        console.error('Mining error:', e);
        alert('エラー: ' + e.message);
        isMining = false;
    }
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

function showResult(nonce, hashArray, elapsed, key, keySeed) {
    let hashHex = Array.from(hashArray).map(b => b.toString(16).padStart(2,'0')).join('');
    const RANDOM_SUFFIX_LEN = 14;
    const randomSuffix = fillRandomBase62(BigInt(keySeed) ^ BigInt(nonce), RANDOM_SUFFIX_LEN);
    const answer = key + randomSuffix;
    const section = document.getElementById('result-section');
    if (section) {
        document.getElementById('result-nonce').textContent = nonce.toLocaleString();
        document.getElementById('result-answer').textContent = answer;
        document.getElementById('result-hash').textContent = hashHex;
        document.getElementById('result-time').textContent = elapsed.toFixed(3) + ' 秒';
        section.classList.remove('hidden');
    }
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
