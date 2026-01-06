#!/usr/bin/env python3
"""
Simple validator to check if a hash matches the target criteria:
- Symmetry = 1
- Efficiency = 1
- Disorder Level = 1
- Order Extracted = 1
"""

import hashlib
import math
import sys
from collections import Counter


def palindromic_hash(data: str, algo="sha256") -> str:
	"""Generate palindromic hash from input data"""
	h = hashlib.new(algo, data.encode()).hexdigest()
	half = h[:len(h)//2]
	return half + half[::-1]


def shannon_entropy(s: str) -> float:
	"""Calculate Shannon entropy of a string"""
	counts = Counter(s)
	total = len(s)
	entropy = 0.0
	for c in counts.values():
		p = c / total
		entropy -= p * math.log2(p)
	return entropy


def ideal_entropy(s: str) -> float:
	"""Calculate ideal entropy based on alphabet size"""
	alphabet = len(set(s))
	if alphabet <= 1:
		return 0.0
	return math.log2(alphabet)


def symmetry_score(s: str) -> float:
	"""Calculate symmetry score (palindrome quality)"""
	matches = sum(1 for i in range(len(s)//2) if s[i] == s[-i-1])
	return matches / (len(s)//2)


def demon_entropy_meter(s: str) -> dict:
	"""Calculate all demon entropy metrics"""
	H_real = shannon_entropy(s)
	H_ideal = ideal_entropy(s)
	symmetry = symmetry_score(s)
	
	entropy_saved = max(0.0, H_ideal - H_real)
	efficiency = entropy_saved / H_ideal if H_ideal > 0 else 0.0
	
	disorder_ratio = H_real / H_ideal if H_ideal > 0 else 0
	order_extracted = 1 - disorder_ratio
	
	return {
		"real_entropy": H_real,
		"ideal_entropy": H_ideal,
		"entropy_saved": entropy_saved,
		"symmetry_score": symmetry,
		"demon_efficiency": efficiency,
		"disorder_level": disorder_ratio,
		"order_extracted": order_extracted
	}


def validate_hash(hash_or_seed: str, tolerance: float = 0.0001) -> dict:
	"""
	Validate if a hash or seed produces metrics matching target criteria.
	
	Args:
		hash_or_seed: Either a palindromic hash or a seed string
		tolerance: Acceptable deviation from target (default 0.0001)
	
	Returns:
		Dictionary with validation results
	"""
	# If it looks like a hex hash, use it directly; otherwise treat as seed
	if len(hash_or_seed) == 64 and all(c in '0123456789abcdef' for c in hash_or_seed.lower()):
		# It's already a hash, make it palindromic if needed
		if hash_or_seed == hash_or_seed[::-1]:
			pal_hash = hash_or_seed
		else:
			half = hash_or_seed[:len(hash_or_seed)//2]
			pal_hash = half + half[::-1]
	else:
		# Treat as seed, generate palindromic hash
		pal_hash = palindromic_hash(hash_or_seed)
	
	# Calculate metrics
	stats = demon_entropy_meter(pal_hash)
	
	# Target values
	targets = {
		"symmetry": 1.0,
		"efficiency": 1.0,
		"disorder": 1.0,
		"order": 1.0
	}
	
	# Check if metrics match targets
	results = {
		"hash": pal_hash,
		"metrics": stats,
		"targets": targets,
		"matches": {},
		"all_match": True,
		"distance": 0.0
	}
	
	# Check each metric
	sym_match = abs(stats["symmetry_score"] - targets["symmetry"]) <= tolerance
	eff_match = abs(stats["demon_efficiency"] - targets["efficiency"]) <= tolerance
	dis_match = abs(stats["disorder_level"] - targets["disorder"]) <= tolerance
	ord_match = abs(stats["order_extracted"] - targets["order"]) <= tolerance
	
	results["matches"] = {
		"symmetry": sym_match,
		"efficiency": eff_match,
		"disorder": dis_match,
		"order": ord_match
	}
	
	# Calculate total distance
	results["distance"] = (
		abs(stats["symmetry_score"] - targets["symmetry"]) +
		abs(stats["demon_efficiency"] - targets["efficiency"]) +
		abs(stats["disorder_level"] - targets["disorder"]) +
		abs(stats["order_extracted"] - targets["order"])
	)
	
	results["all_match"] = sym_match and eff_match and dis_match and ord_match
	
	return results


def print_validation(results: dict):
	"""Print validation results in a readable format"""
	print("\n" + "=" * 70)
	print("VALIDATION RESULTS")
	print("=" * 70)
	print(f"Hash: {results['hash']}")
	print()
	
	print("Metrics vs Targets:")
	print("-" * 70)
	
	metrics = results['metrics']
	matches = results['matches']
	targets = results['targets']
	
	# Symmetry
	sym_status = "✓" if matches["symmetry"] else "✗"
	print(f"{sym_status} Symmetry:      {metrics['symmetry_score']:.10f} (target: {targets['symmetry']:.1f})")
	
	# Efficiency
	eff_status = "✓" if matches["efficiency"] else "✗"
	print(f"{eff_status} Efficiency:    {metrics['demon_efficiency']:.10f} (target: {targets['efficiency']:.1f})")
	
	# Disorder
	dis_status = "✓" if matches["disorder"] else "✗"
	print(f"{dis_status} Disorder:      {metrics['disorder_level']:.10f} (target: {targets['disorder']:.1f})")
	
	# Order
	ord_status = "✓" if matches["order"] else "✗"
	print(f"{ord_status} Order:         {metrics['order_extracted']:.10f} (target: {targets['order']:.1f})")
	
	print("-" * 70)
	print(f"Total Distance: {results['distance']:.10f}")
	print()
	
	if results['all_match']:
		print("🎯 SUCCESS: All metrics match target criteria!")
		print("=" * 70)
		return True
	else:
		print("❌ FAILED: Not all metrics match target criteria")
		print("=" * 70)
		return False


def main():
	"""Main validation function"""
	if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help', 'help']:
		print("Usage: python validate.py <hash_or_seed> [tolerance]")
		print("\nValidates if a hash or seed produces metrics matching target criteria:")
		print("  - Symmetry = 1.0")
		print("  - Efficiency = 1.0")
		print("  - Disorder Level = 1.0")
		print("  - Order Extracted = 1.0")
		print("\nArguments:")
		print("  hash_or_seed: Either a seed string or a 64-char hex hash")
		print("  tolerance:    Acceptable deviation from target (default: 0.0001)")
		print("\nExamples:")
		print("  python validate.py abc123")
		print("  python validate.py 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
		print("  python validate.py abc123 0.01")
		sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help'] else 1)
	
	hash_or_seed = sys.argv[1]
	tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0001
	
	print(f"Validating: {hash_or_seed[:60]}...")
	print(f"Tolerance: {tolerance}")
	
	results = validate_hash(hash_or_seed, tolerance)
	success = print_validation(results)
	
	sys.exit(0 if success else 1)


if __name__ == "__main__":
	main()

