// WebGPU PoW System
let device = null;
let isMining = false;
let stopMining = false;
let startTime = null;
let lastNonce = 0;
let lastTime = null;

// SHA-256 compute shader
const sha256Shader = `
@group(0) @binding(0) var<storage, read> maskBuffer: array<u8>;
@group(0) @binding(1) var<storage, read> keyBuffer: array<u8>;
@group(0) @binding(2) var<storage, read_write> resultBuffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> nonceBuffer: array<u32>;
@group(0) @binding(4) var<storage, read_write> hashBuffer: array<u8>;

// 定数
const RAND_MULT: u64 = 6364136223846793005u;
const RAND_INC: u64 = 1u;
const BASE62: array<u8, 62> = array<u8, 62>(
    48u, 49u, 50u, 51u, 52u, 53u, 54u, 55u, 56u, 57u,
    65u, 66u, 67u, 68u, 69u, 70u, 71u, 72u, 73u, 74u, 75u, 76u, 77u, 78u, 79u, 80u, 81u, 82u, 83u, 84u, 85u, 86u, 87u, 88u, 89u, 90u,
    97u, 98u, 99u, 100u, 101u, 102u, 103u, 104u, 105u, 106u, 107u, 108u, 109u, 110u, 111u, 112u, 113u, 114u, 115u, 116u, 117u, 118u, 119u, 120u, 121u, 122u
);

// SHA-256定数
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

fn sha256_transform(state: ptr<function, array<u32, 8>>, chunk: ptr<function, array<u8, 64>>) {
    var w: array<u32, 64>;
    
    // チャンクをu32配列に変換（ビッグエンディアン）
    for (var i = 0u; i < 16u; i++) {
        let offset = i * 4u;
        w[i] = (u32((*chunk)[offset]) << 24u) |
               (u32((*chunk)[offset + 1u]) << 16u) |
               (u32((*chunk)[offset + 2u]) << 8u) |
               u32((*chunk)[offset + 3u]);
    }
    
    // メッセージスケジュールを拡張
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
    
    // メインループ
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
    let nonce = startNonce + u32(index);
    let keySeed = u64(nonceBuffer[2]) | (u64(nonceBuffer[3]) << 32u);
    let keyLen = u32(nonceBuffer[4]);
    let randomLen = u32(nonceBuffer[5]);
    let maskLen = u32(nonceBuffer[6]);
    let hasNibble = u32(nonceBuffer[7]);
    let maskNibble = u32(nonceBuffer[8]);
    
    // 結果が見つかった場合は早期終了
    if (resultBuffer[0] != 0u) {
        return;
    }
    
    // SHA-256初期化
    var state: array<u32, 8> = array<u32, 8>(
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    );
    
    // メッセージを準備（64バイトのブロック）
    var message: array<u8, 64> = array<u8, 64>(0u);
    
    // キーをコピー
    for (var i = 0u; i < keyLen && i < 64u; i++) {
        message[i] = keyBuffer[i];
    }
    
    // ランダム部分を生成（BASE62）
    var stateRand = keySeed ^ u64(nonce);
    for (var i = 0u; i < randomLen && (keyLen + i) < 64u; i++) {
        stateRand = stateRand * RAND_MULT + RAND_INC;
        let idx = u32((stateRand >> 32u) % 62u);
        message[keyLen + i] = BASE62[idx];
    }
    
    let msgLen = keyLen + randomLen;
    
    // パディング: 0x80を追加
    if (msgLen < 64u) {
        message[msgLen] = 0x80u;
    }
    
    // 長さを追加（ビット長、ビッグエンディアン、最後の8バイト）
    let bitLen = u64(msgLen) * 8u;
    if (msgLen < 56u) {
        // 1ブロックで処理可能
        message[56] = u8((bitLen >> 56u) & 0xffu);
        message[57] = u8((bitLen >> 48u) & 0xffu);
        message[58] = u8((bitLen >> 40u) & 0xffu);
        message[59] = u8((bitLen >> 32u) & 0xffu);
        message[60] = u8((bitLen >> 24u) & 0xffu);
        message[61] = u8((bitLen >> 16u) & 0xffu);
        message[62] = u8((bitLen >> 8u) & 0xffu);
        message[63] = u8(bitLen & 0xffu);
        
        sha256_transform(&state, &message);
    } else {
        // 2ブロック必要
        sha256_transform(&state, &message);
        
        // 2ブロック目を準備
        for (var i = 0u; i < 56u; i++) {
            message[i] = 0u;
        }
        message[56] = u8((bitLen >> 56u) & 0xffu);
        message[57] = u8((bitLen >> 48u) & 0xffu);
        message[58] = u8((bitLen >> 40u) & 0xffu);
        message[59] = u8((bitLen >> 32u) & 0xffu);
        message[60] = u8((bitLen >> 24u) & 0xffu);
        message[61] = u8((bitLen >> 16u) & 0xffu);
        message[62] = u8((bitLen >> 8u) & 0xffu);
        message[63] = u8(bitLen & 0xffu);
        
        sha256_transform(&state, &message);
    }
    
    // ハッシュ結果をバイト配列に変換（ビッグエンディアン）
    var hash: array<u8, 32> = array<u8, 32>(0u);
    for (var i = 0u; i < 8u; i++) {
        hash[i * 4u] = u8((state[i] >> 24u) & 0xffu);
        hash[i * 4u + 1u] = u8((state[i] >> 16u) & 0xffu);
        hash[i * 4u + 2u] = u8((state[i] >> 8u) & 0xffu);
        hash[i * 4u + 3u] = u8(state[i] & 0xffu);
    }
    
    // マスクチェック
    var matches = true;
    for (var i = 0u; i < maskLen && i < 32u; i++) {
        if (hash[i] != maskBuffer[i]) {
            matches = false;
            break;
        }
    }
    
    // 奇数長の場合、最後のニブル（4ビット）のチェック
    if (matches && hasNibble != 0u) {
        if (maskLen >= 32u) {
            matches = false;
        } else {
            let upperNibble = (hash[maskLen] >> 4u) & 0x0fu;
            if (upperNibble != maskNibble) {
                matches = false;
            }
        }
    }
    
    // 結果を保存（アトミック操作を使用）
    if (matches) {
        // atomic compare and exchange
        var expected = 0u;
        var desired = 1u;
        var old = atomicCompareExchangeWeak(&resultBuffer[0], expected, desired);
        if (old == expected) {
            nonceBuffer[1] = nonce;
            for (var i = 0u; i < 32u; i++) {
                hashBuffer[i] = hash[i];
            }
            return;
        }
    }
}
`;

// WebGPUの初期化
async function initWebGPU() {
    const gpuCheck = document.getElementById('gpu-check');
    
    if (!navigator.gpu) {
        gpuCheck.innerHTML = '<p class="error">❌ WebGPUがサポートされていません。このシステムはGPUが必要です。</p>';
        gpuCheck.className = 'gpu-check error';
        return false;
    }
    
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            gpuCheck.innerHTML = '<p class="error">❌ GPUアダプターが見つかりませんでした。GPUが必要です。</p>';
            gpuCheck.className = 'gpu-check error';
            return false;
        }
        
        device = await adapter.requestDevice();
        
        gpuCheck.innerHTML = '<p class="success">✅ GPUが正常に検出されました。計算を開始できます。</p>';
        gpuCheck.className = 'gpu-check success';
        document.getElementById('main-content').classList.remove('hidden');
        return true;
    } catch (error) {
        gpuCheck.innerHTML = `<p class="error">❌ GPUの初期化に失敗しました: ${error.message}</p>`;
        gpuCheck.className = 'gpu-check error';
        return false;
    }
}

// FNV-1a 64ビットハッシュ
function fnv1a64(data) {
    let hash = 0xcbf29ce484222325n;
    for (let i = 0; i < data.length; i++) {
        hash ^= BigInt(data[i]);
        hash = (hash * 0x100000001b3n) & 0xffffffffffffffffn;
    }
    return hash;
}

// マスクをバイト配列に変換（奇数長の場合は最後の文字をニブルとして扱う）
function parseMask(maskStr) {
    const bytes = [];
    let maskNibble = null;
    
    if (maskStr.length % 2 === 1) {
        // 奇数長: 最後の文字をニブルとして扱う
        const fullBytes = maskStr.slice(0, -1);
        const lastChar = maskStr[maskStr.length - 1];
        maskNibble = parseInt(lastChar, 16);
        
        for (let i = 0; i < fullBytes.length; i += 2) {
            const byte = parseInt(fullBytes.substr(i, 2), 16);
            bytes.push(byte);
        }
    } else {
        // 偶数長: 通常通りバイト配列に変換
        for (let i = 0; i < maskStr.length; i += 2) {
            const byte = parseInt(maskStr.substr(i, 2), 16);
            bytes.push(byte);
        }
    }
    
    return { bytes, maskNibble };
}

// BASE62アルファベットからランダム文字列を生成
function fillRandomBase62(state, length) {
    const RAND_MULT = 6364136223846793005n;
    const RAND_INC = 1n;
    const BASE62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
    
    let result = '';
    let currentState = state;
    
    for (let i = 0; i < length; i++) {
        currentState = (currentState * RAND_MULT + RAND_INC) & 0xffffffffffffffffn;
        const idx = Number((currentState >> 32n) % 62n);
        result += BASE62[idx];
    }
    
    return result;
}

// 文字列をバイト配列に変換
function stringToBytes(str) {
    const bytes = [];
    for (let i = 0; i < str.length; i++) {
        bytes.push(str.charCodeAt(i));
    }
    return bytes;
}

// PoW計算の開始
async function startMining(key, mask) {
    if (!device) {
        alert('GPUが初期化されていません');
        return;
    }
    
    isMining = true;
    stopMining = false;
    startTime = Date.now();
    lastNonce = 0;
    lastTime = startTime;
    
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const progressSection = document.getElementById('progress-section');
    const resultSection = document.getElementById('result-section');
    
    startBtn.disabled = true;
    stopBtn.classList.remove('hidden');
    progressSection.classList.remove('hidden');
    resultSection.classList.add('hidden');
    
    try {
        // シェーダーモジュールの作成
        const shaderModule = device.createShaderModule({
            code: sha256Shader
        });
        
        // マスクとキーの準備
        const { bytes: maskBytes, maskNibble } = parseMask(mask);
        const keyBytes = stringToBytes(key);
        const keySeed = fnv1a64(keyBytes);
        const RANDOM_SUFFIX_LEN = 14;
        
        // マスクバッファ（u8配列）- 4の倍数にパディング
        const maskBufferPadded = new Uint8Array(Math.ceil(maskBytes.length / 4) * 4);
        maskBufferPadded.set(maskBytes);
        
        // キーバッファ（u8配列）- 4の倍数にパディング
        const keyBufferPadded = new Uint8Array(Math.ceil(keyBytes.length / 4) * 4);
        keyBufferPadded.set(keyBytes);
        
        // バッファの作成
        const maskGpuBuffer = device.createBuffer({
            size: maskBufferPadded.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        
        const keyGpuBuffer = device.createBuffer({
            size: keyBufferPadded.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        
        const resultBuffer = device.createBuffer({
            size: 4, // u32 (found flag)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        
        const nonceBuffer = device.createBuffer({
            size: 36, // u32[9] (start nonce, found nonce, keySeed low, keySeed high, keyLen, randomLen, maskLen, hasNibble, maskNibble)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        
        const hashBuffer = device.createBuffer({
            size: 32, // u8[32] (hash result)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        
        // バッファにデータを書き込み
        device.queue.writeBuffer(maskGpuBuffer, 0, maskBufferPadded);
        device.queue.writeBuffer(keyGpuBuffer, 0, keyBufferPadded);
        
        // バインドグループの作成
        const bindGroup = device.createBindGroup({
            layout: device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
                ]
            }),
            entries: [
                { binding: 0, resource: { buffer: maskGpuBuffer } },
                { binding: 1, resource: { buffer: keyGpuBuffer } },
                { binding: 2, resource: { buffer: resultBuffer } },
                { binding: 3, resource: { buffer: nonceBuffer } },
                { binding: 4, resource: { buffer: hashBuffer } }
            ]
        });
        
        // 計算パイプラインの作成
        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
        
        // コマンドエンコーダの作成
        let commandEncoder = device.createCommandEncoder();
        let passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        
        // マップされたバッファの作成（結果読み取り用）
        const resultReadBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const nonceReadBuffer = device.createBuffer({
            size: 36,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        const hashReadBuffer = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        let currentNonce = 0;
        const batchSize = 65536; // 256 * 256 workgroups
        
        // nonceバッファの初期データを設定
        const keySeedLow = Number(keySeed & 0xffffffffn);
        const keySeedHigh = Number((keySeed >> 32n) & 0xffffffffn);
        const nonceData = new Uint32Array([
            currentNonce,  // start nonce
            0,             // found nonce
            keySeedLow,    // keySeed low
            keySeedHigh,   // keySeed high
            keyBytes.length, // keyLen
            RANDOM_SUFFIX_LEN, // randomLen
            maskBytes.length, // maskLen
            maskNibble !== null ? 1 : 0, // hasNibble
            maskNibble !== null ? maskNibble : 0 // maskNibble
        ]);
        device.queue.writeBuffer(nonceBuffer, 0, nonceData);
        
        // メインループ
        while (isMining && !stopMining) {
            // 結果バッファをリセット
            const zeroResult = new Uint32Array([0]);
            device.queue.writeBuffer(resultBuffer, 0, zeroResult);
            
            // nonceを更新
            nonceData[0] = currentNonce;
            device.queue.writeBuffer(nonceBuffer, 0, nonceData);
            
            // 計算の実行
            passEncoder.dispatchWorkgroups(Math.ceil(batchSize / 256));
            passEncoder.end();
            
            // 結果をコピー
            commandEncoder.copyBufferToBuffer(resultBuffer, 0, resultReadBuffer, 0, 4);
            commandEncoder.copyBufferToBuffer(nonceBuffer, 0, nonceReadBuffer, 0, 36);
            commandEncoder.copyBufferToBuffer(hashBuffer, 0, hashReadBuffer, 0, 32);
            
            const commandBuffer = commandEncoder.finish();
            device.queue.submit([commandBuffer]);
            
            // 結果を読み取り
            await resultReadBuffer.mapAsync(GPUMapMode.READ);
            await nonceReadBuffer.mapAsync(GPUMapMode.READ);
            
            const resultArray = new Uint32Array(resultReadBuffer.getMappedRange());
            const nonceArray = new Uint32Array(nonceReadBuffer.getMappedRange());
            
            // 進捗の更新
            currentNonce += batchSize;
            lastNonce = currentNonce;
            updateProgress(currentNonce);
            
            if (resultArray[0] !== 0) {
                // 見つかった！
                await hashReadBuffer.mapAsync(GPUMapMode.READ);
                const hashArray = new Uint8Array(hashReadBuffer.getMappedRange());
                
                const foundNonce = nonceArray[1];
                const elapsed = (Date.now() - startTime) / 1000;
                
                // 結果を表示
                showResult(foundNonce, hashArray, elapsed, key, keySeed);
                
                resultReadBuffer.unmap();
                nonceReadBuffer.unmap();
                hashReadBuffer.unmap();
                
                isMining = false;
                break;
            }
            
            resultReadBuffer.unmap();
            nonceReadBuffer.unmap();
            
            // 次のバッチに進む
            currentNonce += batchSize;
            
            // 新しいコマンドエンコーダを作成（次のループ用）
            commandEncoder = device.createCommandEncoder();
            passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup);
        }
        
        startBtn.disabled = false;
        stopBtn.classList.add('hidden');
        
    } catch (error) {
        console.error('Mining error:', error);
        alert(`計算中にエラーが発生しました: ${error.message}`);
        isMining = false;
        startBtn.disabled = false;
        stopBtn.classList.add('hidden');
    }
}

// 進捗の更新
function updateProgress(nonce) {
    const currentTime = Date.now();
    const elapsed = (currentTime - startTime) / 1000;
    
    // ハッシュレートの計算
    let hashRate = 0;
    if (elapsed > 0) {
        hashRate = Math.floor(nonce / elapsed);
    }
    
    // UI更新
    document.getElementById('current-nonce').textContent = nonce.toLocaleString();
    document.getElementById('hash-rate').textContent = hashRate.toLocaleString();
    document.getElementById('elapsed-time').textContent = elapsed.toFixed(2);
    
    // プログレスバー（最大値は仮に設定）
    const maxNonce = 10000000;
    const progress = Math.min((nonce / maxNonce) * 100, 100);
    document.getElementById('progress-bar').style.width = progress + '%';
}

// 結果の表示
function showResult(nonce, hashArray, elapsed, key, keySeed) {
    // ハッシュを16進数文字列に変換
    let hashHex = '';
    for (let i = 0; i < hashArray.length; i++) {
        hashHex += hashArray[i].toString(16).padStart(2, '0');
    }
    
    // Answerの生成（key + ランダムサフィックス）
    const RANDOM_SUFFIX_LEN = 14;
    const randomSuffix = fillRandomBase62(keySeed ^ BigInt(nonce), RANDOM_SUFFIX_LEN);
    const answer = key + randomSuffix;
    
    document.getElementById('result-nonce').textContent = nonce.toLocaleString();
    document.getElementById('result-answer').textContent = answer;
    document.getElementById('result-hash').textContent = hashHex;
    document.getElementById('result-time').textContent = elapsed.toFixed(3) + ' 秒';
    
    document.getElementById('result-section').classList.remove('hidden');
}

// イベントリスナーの設定
document.addEventListener('DOMContentLoaded', async () => {
    const gpuInitialized = await initWebGPU();
    
    if (gpuInitialized) {
        const form = document.getElementById('pow-form');
        const stopBtn = document.getElementById('stop-btn');
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const key = document.getElementById('key-input').value.trim();
            const mask = document.getElementById('mask-input').value.trim();
            
            if (key.length !== 2) {
                alert('Keyは2文字である必要があります');
                return;
            }
            
            if (!/^[0-9a-fA-F]+$/.test(mask)) {
                alert('Maskは16進数文字列である必要があります');
                return;
            }
            
            await startMining(key, mask);
        });
        
        stopBtn.addEventListener('click', () => {
            stopMining = true;
            isMining = false;
        });
    }
});

