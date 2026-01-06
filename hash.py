import hashlib
import time
import math
from collections import Counter
from dataclasses import dataclass, asdict


# -----------------------------
# Palindromic hash
# -----------------------------
def palindromic_hash(data: str, algo="sha256") -> str:
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
# Run
# -----------------------------
if __name__ == "__main__":
	run_demon_chain(delay=1.2)
