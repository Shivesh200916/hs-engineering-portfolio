import math

def compute_euler_totient(n):
    """Calculate Euler's totient function phi(n) using prime factorization."""
    result = n
    p = 2
    temp_n = n
    while p * p <= temp_n:
        if temp_n % p == 0:
            while temp_n % p == 0:
                temp_n //= p
            result -= result // p
        p += 1
    if temp_n > 1:
        result -= result // temp_n
    return result

def search_lehmer_counterexamples(limit):
    """
    Searches for composite numbers where phi(n) divides (n - 1).
    Applies optimization techniques by pruning known impossible constraints.
    """
    print(f"Starting optimized scan up to: {limit:,}")
    found_counterexamples = []
    
    # Optimization: Lehmer proved any counterexample must be odd and square-free.
    # Therefore, we skip all even numbers entirely.
    for n in range(3, limit, 2):
        # Quick prime check: Primes satisfy phi(n) = n - 1 trivially.
        # We are strictly looking for composite numbers.
        if is_prime(n):
            continue
            
        phi = compute_euler_totient(n)
        
        # Core mathematical constraint check
        if (n - 1) % phi == 0:
            print(f"⚠️ Found potential anomaly/counterexample: {n}")
            found_counterexamples.append(n)
            
    print("Scan complete.")
    return found_counterexamples

def is_prime(n):
    """Helper function for fast primality verification."""
    if n < 2: return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0: return False
    return True

if __name__ == "__main__":
    # Run a test scan across a small sample bounds to demonstrate backend execution
    search_lehmer_counterexamples(limit=50000)
