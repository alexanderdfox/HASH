// Web Worker for parallel hash computation
// This runs in a separate thread

// Hash functions
async function sha256Hash(data) {
    const encoder = new TextEncoder();
    const dataBuffer = encoder.encode(data);
    const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

function palindromicHash(hexHash) {
    const half = hexHash.substring(0, Math.floor(hexHash.length / 2));
    return half + half.split('').reverse().join('');
}

function shannonEntropy(s) {
    const counts = {};
    for (let char of s) {
        counts[char] = (counts[char] || 0) + 1;
    }
    
    let entropy = 0;
    const total = s.length;
    for (let count of Object.values(counts)) {
        const p = count / total;
        entropy -= p * Math.log2(p);
    }
    return entropy;
}

function idealEntropy(s) {
    const alphabet = new Set(s).size;
    if (alphabet <= 1) return 0;
    return Math.log2(alphabet);
}

function symmetryScore(s) {
    let matches = 0;
    const half = Math.floor(s.length / 2);
    for (let i = 0; i < half; i++) {
        if (s[i] === s[s.length - 1 - i]) {
            matches++;
        }
    }
    return matches / half;
}

function demonEntropyMeter(s) {
    const H_real = shannonEntropy(s);
    const H_ideal = idealEntropy(s);
    const symmetry = symmetryScore(s);
    
    const entropy_saved = Math.max(0, H_ideal - H_real);
    const efficiency = H_ideal > 0 ? entropy_saved / H_ideal : 0;
    
    const disorder_ratio = H_ideal > 0 ? H_real / H_ideal : 0;
    const order_extracted = 1 - disorder_ratio;
    
    return {
        real_entropy: H_real,
        ideal_entropy: H_ideal,
        entropy_saved: entropy_saved,
        symmetry_score: symmetry,
        demon_efficiency: efficiency,
        disorder_level: disorder_ratio,
        order_extracted: order_extracted
    };
}

function calculateDistance(stats, targets) {
    const symDiff = Math.abs(stats.symmetry_score - targets.symmetry);
    const effDiff = Math.abs(stats.demon_efficiency - targets.efficiency);
    const disDiff = Math.abs(stats.disorder_level - targets.disorder);
    const ordDiff = Math.abs(stats.order_extracted - targets.order);
    return symDiff + effDiff + disDiff + ordDiff;
}

// Listen for messages from main thread
self.addEventListener('message', async function(e) {
    const { seeds, targets, workerId } = e.data;
    
    const results = [];
    
    for (const seed of seeds) {
        try {
            const hash = await sha256Hash(seed);
            const palHash = palindromicHash(hash);
            const stats = demonEntropyMeter(palHash);
            const distance = calculateDistance(stats, targets);
            
            results.push({
                seed: seed,
                hash: palHash,
                stats: stats,
                distance: distance
            });
        } catch (error) {
            console.error(`Worker ${workerId} error:`, error);
        }
    }
    
    // Send results back to main thread
    self.postMessage({
        workerId: workerId,
        results: results
    });
});

