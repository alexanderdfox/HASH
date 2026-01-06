# Maxwell's Demon Blockchain - GPU & Security Enclave Enhanced

A palindromic blockchain analyzer with GPU acceleration and security enclave support. This project implements Maxwell's Demon concept through entropy analysis of palindromic hashes, with both web and Python implementations.

## Features

### 🚀 GPU Acceleration
- **HTML/WebGPU**: Parallel hash computation using WebGPU for massive performance gains
- **Python/PyTorch**: CUDA-accelerated batch processing for hash searches
- Automatic fallback to CPU if GPU is unavailable
- Multi-threaded processing support (Python)

### 🔒 Security Enclave
- **HTML**: Web Crypto API with secure contexts for encrypted hash operations
- **Python**: Secure memory handling with cryptographically secure random number generation
- Isolated secure memory for sensitive operations
- AES-GCM encryption in web version

### 📊 Hash Search & Validation
- Search for hashes matching target entropy metrics
- Real-time progress tracking and visualization
- Validation tool to verify hash metrics
- Multi-threaded and GPU-accelerated search modes

## Project Structure

```
HASH/
├── hash.py              # Main Python implementation
├── hash-worker.js       # Web Worker for browser parallel processing
├── index.html           # Web interface with Matrix theme
├── validate.py          # Hash validation tool
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

## Usage

### HTML Version

1. Open `index.html` in a modern browser (Chrome/Edge with WebGPU support)
2. The system will automatically detect and initialize:
   - GPU acceleration (WebGPU)
   - Security Enclave (Web Crypto API)
   - Web Workers for multi-threading
3. Set your target values (Symmetry, Efficiency, Disorder Level, Order Extracted) and click "INITIATE SEARCH"
4. Watch the real-time search with Matrix-style visualization
5. The search will automatically stop when a perfect match (all metrics = 1.0) is found

**System Requirements:**
- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- HTTPS or localhost for security enclave features
- JavaScript enabled

**Features:**
- Real-time closest match display
- Progress tracking with iteration count
- Visual notifications for matches
- Automatic worker management based on CPU cores

### Python Version

#### Basic Usage
```bash
# Normal blockchain mode (continuous chain generation)
python hash.py

# Search mode (CPU, single-threaded)
python hash.py search

# Search mode with GPU acceleration
python hash.py search --gpu
# or
python hash.py search -g

# Search mode with Security Enclave
python hash.py search --enclave
# or
python hash.py search -e

# Search mode with both GPU and Enclave
python hash.py search --gpu --enclave

# Multi-threaded search (auto-detects CPU cores)
python hash.py search --threads=8
# or
python hash.py search -t 8

# Combined: GPU + Enclave + Multi-threading
python hash.py search --gpu --enclave --threads=4
```

#### Installation

**Basic (CPU only):**
```bash
# No dependencies required - uses standard library only
python hash.py
```

**With GPU Support:**
```bash
pip install torch
python hash.py search --gpu
```

**Install from requirements.txt:**
```bash
pip install -r requirements.txt
```

### Validation Tool

Use `validate.py` to verify if a hash or seed matches the target criteria:

```bash
# Validate a seed string
python validate.py "my-seed-string"

# Validate a hash directly
python validate.py "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

# Use custom tolerance
python validate.py "my-seed" 0.01

# Show help
python validate.py --help
```

The validator checks if all four metrics equal 1.0:
- Symmetry Score = 1.0
- Demon Efficiency = 1.0
- Disorder Level = 1.0
- Order Extracted = 1.0

## Performance

- **CPU Mode (Single-threaded)**: ~1,000 hashes/second
- **CPU Mode (Multi-threaded)**: ~1,000 × N hashes/second (N = number of threads)
- **GPU Mode**: ~10,000-100,000 hashes/second (depending on GPU)
- **Batch Processing**: Processes 64 hashes in parallel when GPU is enabled
- **Web Workers**: Automatically uses all available CPU cores in browser

## Security Features

### Security Enclave
- Cryptographically secure random number generation (`secrets` module in Python, Web Crypto API in browser)
- Encrypted hash operations (AES-GCM in web version)
- Secure memory isolation
- Protection against timing attacks
- Secure random salt generation

### Web Version Security
- Requires secure context (HTTPS or localhost)
- Uses Web Crypto API for all cryptographic operations
- Secure random number generation
- Memory isolation
- Encrypted data handling

## Architecture

### GPU Acceleration
- **WebGPU**: Compute shaders for parallel hash computation
- **PyTorch**: CUDA tensors for batch entropy calculations
- Automatic batch processing for optimal performance
- Graceful fallback to CPU if GPU unavailable

### Security Enclave
- **Web**: Web Crypto API with AES-GCM encryption
- **Python**: Secure random number generation with `secrets` module
- Isolated secure memory storage
- Salt-based hash protection

### Multi-threading
- **Python**: Threading support with configurable thread count
- **Web**: Web Workers automatically spawn based on CPU cores
- Parallel search across multiple threads/workers
- Result aggregation and best match tracking

## Metrics Explained

The system calculates four key metrics for each palindromic hash:

1. **Symmetry Score**: Measures how well the hash matches its palindrome structure (0-1.0)
2. **Demon Efficiency**: Ratio of entropy saved to ideal entropy (0-1.0)
3. **Disorder Level**: Ratio of real entropy to ideal entropy (0-1.0)
4. **Order Extracted**: Complement of disorder level (1 - disorder) (0-1.0)

**Note**: Having all four metrics equal 1.0 simultaneously is mathematically impossible due to the relationship between entropy and symmetry. The search finds the closest possible match.

## Matrix Theme

The HTML interface features:
- Green-on-black Matrix aesthetic
- Falling character rain animation (Japanese katakana and binary)
- Real-time metrics visualization with progress bars
- Glowing text effects and animations
- Responsive design
- Visual notifications for matches
- Closest match highlighting

## Command-Line Options (Python)

| Option | Short | Description |
|--------|-------|-------------|
| `search` | - | Enable search mode (default: blockchain mode) |
| `--gpu` | `-g` | Enable GPU acceleration |
| `--enclave` | `-e` | Enable security enclave |
| `--threads=N` | `-t N` | Use N threads (default: auto-detect CPU cores) |

## Notes

- Finding hashes with all metrics = 1.0 simultaneously is mathematically impossible
- The search finds the closest possible match to target values
- GPU acceleration provides 10-100x speedup depending on hardware
- Security enclave adds minimal overhead (~5-10%) for enhanced security
- Multi-threading provides near-linear speedup on multi-core CPUs
- Web version automatically uses all available CPU cores via Web Workers

## Browser Compatibility

- **WebGPU**: Chrome 113+, Edge 113+, Firefox (experimental)
- **Web Crypto API**: All modern browsers
- **Web Workers**: All modern browsers
- **Secure Context**: HTTPS or localhost required for security features

## Dependencies

### Python
- **Required**: None (uses standard library)
- **Optional**: `torch>=2.0.0` (for GPU acceleration)

### Web
- No external dependencies (uses browser APIs only)

## License

MIT License - Copyright (c) 2026 Alex Fox

Feel free to use and modify!

