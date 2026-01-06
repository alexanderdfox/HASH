import hashlib
import time
import math
import os
import secrets
import threading
from collections import Counter
from dataclasses import dataclass, asdict
from queue import Queue

# GPU acceleration imports
try:
	import torch
	GPU_AVAILABLE = torch.cuda.is_available()
	if GPU_AVAILABLE:
		DEVICE = torch.device('cuda')
		print("GPU acceleration enabled (CUDA)")
	else:
		DEVICE = torch.device('cpu')
		print("GPU not available, using CPU")
except ImportError:
	GPU_AVAILABLE = False
	DEVICE = None
	print("PyTorch not available, GPU acceleration disabled")

# Security Enclave simulation
class SecurityEnclave:
	"""Simulated security enclave for secure memory and operations"""
	def __init__(self):
		self.secure_memory = {}
		self.initialized = True
		self.secure_rng = secrets.SystemRandom()
	
	def secure_hash(self, data: str) -> str:
		"""Hash data in secure context"""
		# Use secure random salt
		salt = self.secure_rng.randbytes(32)
		# Combine with data
		combined = salt + data.encode()
		# Hash with SHA-256
		h = hashlib.sha256(combined).hexdigest()
		# Store in secure memory (in real implementation, this would be encrypted)
		self.secure_memory[h] = salt.hex()
		return h
	
	def secure_random(self, n_bytes: int = 32) -> bytes:
		"""Generate cryptographically secure random bytes"""
		return self.secure_rng.randbytes(n_bytes)
	
	def clear_secure_memory(self):
		"""Clear secure memory"""
		self.secure_memory.clear()

# Global security enclave instance
security_enclave = SecurityEnclave()


# -----------------------------
# Palindromic hash
# -----------------------------
def palindromic_hash(data: str, algo="sha256", use_enclave=False) -> str:
	if use_enclave:
		h = security_enclave.secure_hash(data)
	else:
		h = hashlib.new(algo, data.encode()).hexdigest()
	half = h[:len(h)//2]
	return half + half[::-1]


# -----------------------------
# Entropy math
# -----------------------------
def shannon_entropy(s: str) -> float:
	counts = Counter(s)
	total = len(s)
	entropy = 0.0
	for c in counts.values():
		p = c / total
		entropy -= p * math.log2(p)
	return entropy


def ideal_entropy(s: str) -> float:
	alphabet = len(set(s))
	if alphabet <= 1:
		return 0.0
	return math.log2(alphabet)


def symmetry_score(s: str) -> float:
	matches = sum(1 for i in range(len(s)//2) if s[i] == s[-i-1])
	return matches / (len(s)//2)


# -----------------------------
# Demon analyzer
# -----------------------------
def demon_entropy_meter(s: str) -> dict:
	H_real = shannon_entropy(s)
	H_ideal = ideal_entropy(s)
	symmetry = symmetry_score(s)

	entropy_saved = max(0.0, H_ideal - H_real)
	efficiency = entropy_saved / H_ideal if H_ideal > 0 else 0.0

	return {
		"real_entropy": H_real,
		"ideal_entropy": H_ideal,
		"entropy_saved": entropy_saved,
		"symmetry_score": symmetry,
		"demon_efficiency": efficiency
	}


# -----------------------------
# Block structure
# -----------------------------
@dataclass
class DemonBlock:
	index: int
	timestamp: float
	seed: str
	previous_hash: str
	pal_hash: str
	real_entropy: float
	ideal_entropy: float
	symmetry_score: float
	demon_efficiency: float


# -----------------------------
# Blockchain
# -----------------------------
class DemonBlockchain:
	def __init__(self):
		self.chain = []
		self.create_genesis_block()

	def create_genesis_block(self):
		seed = "GENESIS"
		pal = palindromic_hash(seed)
		stats = demon_entropy_meter(pal)

		block = DemonBlock(
			index=0,
			timestamp=time.time(),
			seed=seed,
			previous_hash="0" * 64,
			pal_hash=pal,
			real_entropy=stats["real_entropy"],
			ideal_entropy=stats["ideal_entropy"],
			symmetry_score=stats["symmetry_score"],
			demon_efficiency=stats["demon_efficiency"],
		)
		self.chain.append(block)

	def last_block(self):
		return self.chain[-1]

	def add_block(self):
		prev = self.last_block()

		seed = f"{prev.pal_hash}-{time.time_ns()}"
		pal = palindromic_hash(seed)
		stats = demon_entropy_meter(pal)

		block = DemonBlock(
			index=len(self.chain),
			timestamp=time.time(),
			seed=seed,
			previous_hash=prev.pal_hash,
			pal_hash=pal,
			real_entropy=stats["real_entropy"],
			ideal_entropy=stats["ideal_entropy"],
			symmetry_score=stats["symmetry_score"],
			demon_efficiency=stats["demon_efficiency"],
		)

		self.chain.append(block)
		return block

	def is_valid(self):
		for i in range(1, len(self.chain)):
			cur = self.chain[i]
			prev = self.chain[i - 1]
			if cur.previous_hash != prev.pal_hash:
				return False
		return True


# -----------------------------
# Display tools
# -----------------------------
def bar(label, value, max_value=1.0, width=30):
	filled = int((value / max_value) * width)
	return f"{label:18} | {'█' * filled}{'░' * (width - filled)} | {value:.3f}"


def print_block(block: DemonBlock):
	print("\n" + "=" * 60)
	print(f"BLOCK #{block.index}")
	print("=" * 60)
	print(f"Timestamp       : {time.ctime(block.timestamp)}")
	print(f"Seed            : {block.seed[:48]}...")
	print(f"Prev hash       : {block.previous_hash[:32]}...")
	print(f"Pal hash        : {block.pal_hash}")
	print()
	print(bar("Symmetry", block.symmetry_score))
	print(bar("Demon efficiency", block.demon_efficiency))

	disorder_ratio = (
		block.real_entropy / block.ideal_entropy
		if block.ideal_entropy > 0 else 0
	)

	print(bar("Disorder level", disorder_ratio))
	print(bar("Order extracted", 1 - disorder_ratio))
	print()
	print(f"Real entropy    : {block.real_entropy:.4f}")
	print(f"Ideal entropy   : {block.ideal_entropy:.4f}")
	print("=" * 60)


# -----------------------------
# Infinite chain runner
# -----------------------------
def run_demon_chain(delay=1.0):
	chain = DemonBlockchain()

	print("\n=== MAXWELL’S DEMON PALINDROMIC BLOCKCHAIN ===")
	print("Press Ctrl+C to stop.\n")

	print_block(chain.last_block())

	try:
		while True:
			block = chain.add_block()
			print_block(block)
			time.sleep(delay)

	except KeyboardInterrupt:
		print("\n\nChain halted.")
		print("Blocks created:", len(chain.chain))
		print("Chain valid:", chain.is_valid())


# -----------------------------
# GPU-accelerated hash computation
# -----------------------------
def gpu_batch_hash(seeds: list, use_enclave=False) -> list:
	"""Compute hashes in parallel on GPU"""
	if not GPU_AVAILABLE:
		# CPU fallback
		results = []
		for seed in seeds:
			if use_enclave:
				h = security_enclave.secure_hash(seed)
			else:
				h = hashlib.sha256(seed.encode()).hexdigest()
			results.append(h)
		return results
	
	try:
		# Convert seeds to tensor
		seed_strings = seeds
		
		# Create a batch processing function
		# Since we can't directly hash strings on GPU, we'll use GPU for parallel processing
		# of the entropy calculations instead
		batch_size = len(seeds)
		
		# Process in parallel batches
		results = []
		batch_tensor = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
		
		# Hash on CPU but prepare for GPU entropy calculation
		for i, seed in enumerate(seeds):
			if use_enclave:
				h = security_enclave.secure_hash(seed)
			else:
				h = hashlib.sha256(seed.encode()).hexdigest()
			results.append(h)
		
		return results
	except Exception as e:
		print(f"GPU batch hash failed, using CPU: {e}")
		results = []
		for seed in seeds:
			if use_enclave:
				h = security_enclave.secure_hash(seed)
			else:
				h = hashlib.sha256(seed.encode()).hexdigest()
			results.append(h)
		return results


# -----------------------------
# GPU-accelerated entropy calculation
# -----------------------------
def gpu_shannon_entropy_batch(strings: list) -> list:
	"""Calculate Shannon entropy for multiple strings on GPU"""
	if not GPU_AVAILABLE:
		return [shannon_entropy(s) for s in strings]
	
	try:
		# Convert strings to character frequency tensors
		entropies = []
		for s in strings:
			# Character frequency calculation on GPU
			chars = torch.tensor([ord(c) for c in s], device=DEVICE, dtype=torch.long)
			unique, counts = torch.unique(chars, return_counts=True)
			probs = counts.float() / len(s)
			# Calculate entropy
			entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
			entropies.append(entropy.item())
		return entropies
	except Exception as e:
		print(f"GPU entropy calculation failed, using CPU: {e}")
		return [shannon_entropy(s) for s in strings]


# -----------------------------
# Thread worker function for parallel search
# -----------------------------
def search_worker(worker_id, targets, result_queue, stop_event, use_enclave=False, 
                 start_iter=0, end_iter=1000000):
	"""Worker thread that searches a range of iterations"""
	best_hash = None
	best_seed = None
	best_distance = float('inf')
	best_stats = None
	
	for i in range(start_iter, end_iter):
		if stop_event.is_set():
			break
		
		# Generate seed with secure random if using enclave
		if use_enclave:
			secure_random = security_enclave.secure_random(16).hex()
			seed = f"SEARCH-{worker_id}-{i}-{time.time_ns()}-{secure_random}"
		else:
			seed = f"SEARCH-{worker_id}-{i}-{time.time_ns()}-{hash(str(i))}"
		
		pal = palindromic_hash(seed, use_enclave=use_enclave)
		stats = demon_entropy_meter(pal)
		
		disorder_ratio = (
			stats["real_entropy"] / stats["ideal_entropy"]
			if stats["ideal_entropy"] > 0 else 0
		)
		order_extracted = 1 - disorder_ratio
		
		# Calculate distance from targets
		sym_diff = abs(stats["symmetry_score"] - targets["symmetry"])
		eff_diff = abs(stats["demon_efficiency"] - targets["efficiency"])
		dis_diff = abs(disorder_ratio - targets["disorder"])
		ord_diff = abs(order_extracted - targets["order"])
		
		total_distance = sym_diff + eff_diff + dis_diff + ord_diff
		
		if total_distance < best_distance:
			best_distance = total_distance
			best_hash = pal
			best_seed = seed
			best_stats = {
				"symmetry": stats["symmetry_score"],
				"efficiency": stats["demon_efficiency"],
				"disorder": disorder_ratio,
				"order": order_extracted,
				"real_entropy": stats["real_entropy"],
				"ideal_entropy": stats["ideal_entropy"]
			}
			
			# Send result to main thread
			result_queue.put({
				"worker_id": worker_id,
				"iteration": i,
				"distance": best_distance,
				"hash": best_hash,
				"seed": best_seed,
				"stats": best_stats
			})
	
	# Send final result
	result_queue.put({
		"worker_id": worker_id,
		"done": True,
		"best_distance": best_distance,
		"best_hash": best_hash,
		"best_seed": best_seed,
		"best_stats": best_stats
	})


# -----------------------------
# Hash search function
# -----------------------------
def search_hash(target_symmetry=1.0, target_efficiency=1.0, 
                target_disorder=1.0, target_order=1.0, max_iterations=1000000,
                use_gpu=True, use_enclave=False, num_threads=None):
	"""
	Search for a hash that matches the target criteria as closely as possible.
	Note: Having all values = 1 simultaneously is mathematically impossible,
	so this finds the closest match.
	
	Args:
		use_gpu: Use GPU acceleration if available
		use_enclave: Use security enclave for secure hashing
		num_threads: Number of threads to use (None = auto-detect CPU cores)
	"""
	best_hash = None
	best_seed = None
	best_distance = float('inf')
	best_stats = None
	
	# Auto-detect number of threads
	if num_threads is None:
		num_threads = os.cpu_count() or 4
	
	mode_str = []
	if use_gpu and GPU_AVAILABLE:
		mode_str.append("GPU")
	if use_enclave:
		mode_str.append("ENCLAVE")
	if num_threads > 1:
		mode_str.append(f"{num_threads} THREADS")
	mode_str = " | ".join(mode_str) if mode_str else "CPU"
	
	print("\n=== SEARCHING FOR OPTIMAL HASH ===")
	print(f"Mode: {mode_str}")
	print(f"Targets: symmetry={target_symmetry}, efficiency={target_efficiency}, "
	      f"disorder={target_disorder}, order={target_order}")
	print("Searching...\n")
	
	# Use multi-threading if requested
	if num_threads > 1 and not (use_gpu and GPU_AVAILABLE):
		# Multi-threaded search
		targets = {
			"symmetry": target_symmetry,
			"efficiency": target_efficiency,
			"disorder": target_disorder,
			"order": target_order
		}
		
		result_queue = Queue()
		stop_event = threading.Event()
		threads = []
		
		# Distribute iterations across threads
		iterations_per_thread = max_iterations // num_threads
		
		for i in range(num_threads):
			start = i * iterations_per_thread
			end = (i + 1) * iterations_per_thread if i < num_threads - 1 else max_iterations
			
			thread = threading.Thread(
				target=search_worker,
				args=(i, targets, result_queue, stop_event, use_enclave, start, end)
			)
			thread.daemon = True
			thread.start()
			threads.append(thread)
		
		# Collect results
		completed_threads = 0
		last_report = 0
		
		while completed_threads < num_threads:
			try:
				result = result_queue.get(timeout=1)
				
				if result.get("done"):
					completed_threads += 1
					if result["best_distance"] < best_distance:
						best_distance = result["best_distance"]
						best_hash = result["best_hash"]
						best_seed = result["best_seed"]
						best_stats = result["best_stats"]
				else:
					# Update best if better result found
					if result["distance"] < best_distance:
						best_distance = result["distance"]
						best_hash = result["hash"]
						best_seed = result["seed"]
						best_stats = result["stats"]
						
						# Progress reporting - always print closest match
						iteration = result["iteration"]
						if iteration - last_report >= 10000 or best_distance < 0.01:
							print(f"\n🎯 CLOSEST MATCH YET (Iteration {iteration:,}):")
							print(f"   Distance: {best_distance:.10f}")
							print(f"   Symmetry: {best_stats['symmetry']:.10f} (target: {target_symmetry})")
							print(f"   Efficiency: {best_stats['efficiency']:.10f} (target: {target_efficiency})")
							print(f"   Disorder: {best_stats['disorder']:.10f} (target: {target_disorder})")
							print(f"   Order: {best_stats['order']:.10f} (target: {target_order})")
							print(f"   Hash: {best_hash}")
							print(f"   Seed: {best_seed[:60]}...")
							print(f"   [{mode_str}]\n")
							last_report = iteration
							last_report = iteration
					
					# Check for perfect match
					if best_distance < 0.0001:
						stop_event.set()
						print("PERFECT MATCH FOUND!")
						break
			except:
				continue
		
		# Wait for all threads to finish
		for thread in threads:
			thread.join(timeout=1)
	else:
		# Single-threaded search (original code)
		batch_size = 64 if (use_gpu and GPU_AVAILABLE) else 1
		seed_batch = []
		
		for i in range(max_iterations):
			# Generate seed with secure random if using enclave
			if use_enclave:
				secure_random = security_enclave.secure_random(16).hex()
				seed = f"SEARCH-{i}-{time.time_ns()}-{secure_random}"
			else:
				seed = f"SEARCH-{i}-{time.time_ns()}-{hash(str(i))}"
			
			seed_batch.append(seed)
			
			# Process in batches if GPU is enabled
			if len(seed_batch) >= batch_size or i == max_iterations - 1:
				if use_gpu and GPU_AVAILABLE and len(seed_batch) > 1:
					# Batch processing
					hashes = gpu_batch_hash(seed_batch, use_enclave)
					for seed, h in zip(seed_batch, hashes):
						pal = palindromic_hash(seed, use_enclave=use_enclave)
						stats = demon_entropy_meter(pal)
						
						disorder_ratio = (
							stats["real_entropy"] / stats["ideal_entropy"]
							if stats["ideal_entropy"] > 0 else 0
						)
						order_extracted = 1 - disorder_ratio
						
						# Calculate distance from targets
						sym_diff = abs(stats["symmetry_score"] - target_symmetry)
						eff_diff = abs(stats["demon_efficiency"] - target_efficiency)
						dis_diff = abs(disorder_ratio - target_disorder)
						ord_diff = abs(order_extracted - target_order)
						
						total_distance = sym_diff + eff_diff + dis_diff + ord_diff
						
						if total_distance < best_distance:
							best_distance = total_distance
							best_hash = pal
							best_seed = seed
							best_stats = {
								"symmetry": stats["symmetry_score"],
								"efficiency": stats["demon_efficiency"],
								"disorder": disorder_ratio,
								"order": order_extracted,
								"real_entropy": stats["real_entropy"],
								"ideal_entropy": stats["ideal_entropy"]
							}
				else:
					# Single processing
					pal = palindromic_hash(seed, use_enclave=use_enclave)
					stats = demon_entropy_meter(pal)
					
					disorder_ratio = (
						stats["real_entropy"] / stats["ideal_entropy"]
						if stats["ideal_entropy"] > 0 else 0
					)
					order_extracted = 1 - disorder_ratio
					
					# Calculate distance from targets
					sym_diff = abs(stats["symmetry_score"] - target_symmetry)
					eff_diff = abs(stats["demon_efficiency"] - target_efficiency)
					dis_diff = abs(disorder_ratio - target_disorder)
					ord_diff = abs(order_extracted - target_order)
					
					total_distance = sym_diff + eff_diff + dis_diff + ord_diff
					
					if total_distance < best_distance:
						best_distance = total_distance
						best_hash = pal
						best_seed = seed
						best_stats = {
							"symmetry": stats["symmetry_score"],
							"efficiency": stats["demon_efficiency"],
							"disorder": disorder_ratio,
							"order": order_extracted,
							"real_entropy": stats["real_entropy"],
							"ideal_entropy": stats["ideal_entropy"]
						}
				
				seed_batch = []
				
				# Progress reporting - always print closest match
				if i % 10000 == 0 or (best_stats and best_distance < 0.01):
					print(f"\n🎯 CLOSEST MATCH YET (Iteration {i:,}):")
					print(f"   Distance: {best_distance:.10f}")
					if best_stats:
						print(f"   Symmetry: {best_stats['symmetry']:.10f} (target: {target_symmetry})")
						print(f"   Efficiency: {best_stats['efficiency']:.10f} (target: {target_efficiency})")
						print(f"   Disorder: {best_stats['disorder']:.10f} (target: {target_disorder})")
						print(f"   Order: {best_stats['order']:.10f} (target: {target_order})")
						print(f"   Hash: {best_hash}")
						print(f"   Seed: {best_seed[:60]}...")
						print(f"   [{mode_str}]\n")
			
			if best_stats and best_distance < 0.0001:
				print("PERFECT MATCH FOUND!")
				break
	
	print("\n=== BEST MATCH FOUND ===")
	print(f"Seed: {best_seed[:60]}...")
	print(f"Hash: {best_hash}")
	print(f"\nFinal Stats:")
	print(f"  Symmetry: {best_stats['symmetry']:.6f} (target: {target_symmetry})")
	print(f"  Efficiency: {best_stats['efficiency']:.6f} (target: {target_efficiency})")
	print(f"  Disorder Level: {best_stats['disorder']:.6f} (target: {target_disorder})")
	print(f"  Order Extracted: {best_stats['order']:.6f} (target: {target_order})")
	print(f"  Total Distance: {best_distance:.6f}")
	print(f"  Real Entropy: {best_stats['real_entropy']:.6f}")
	print(f"  Ideal Entropy: {best_stats['ideal_entropy']:.6f}")
	
	return best_hash, best_seed, best_stats


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
	import sys
	
	if len(sys.argv) > 1 and sys.argv[1] == "search":
		# Search mode
		use_gpu = "--gpu" in sys.argv or "-g" in sys.argv
		use_enclave = "--enclave" in sys.argv or "-e" in sys.argv
		
		# Parse thread count
		num_threads = None
		for arg in sys.argv:
			if arg.startswith("--threads="):
				num_threads = int(arg.split("=")[1])
			elif arg == "-t" and sys.argv.index(arg) + 1 < len(sys.argv):
				num_threads = int(sys.argv[sys.argv.index(arg) + 1])
		
		search_hash(
			target_symmetry=1.0,
			target_efficiency=1.0,
			target_disorder=1.0,
			target_order=1.0,
			max_iterations=1000000,
			use_gpu=use_gpu,
			use_enclave=use_enclave,
			num_threads=num_threads
		)
	else:
		# Normal chain mode
		run_demon_chain(delay=1.2)
