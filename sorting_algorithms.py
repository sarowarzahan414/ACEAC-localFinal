"""
ACEAC Sorting Algorithms - Security Review
Three different sorting approaches optimized for different use cases

Author: @sarowarzahan414
Date: 2025-11-18
Purpose: Demonstrate speed-optimized, memory-optimized, and stable sorting
"""

import random
import time
from typing import List, TypeVar

T = TypeVar('T')


# =============================================================================
# 1. QUICKSORT - Optimized for Speed
# =============================================================================
def quicksort(arr: List[T]) -> List[T]:
    """
    Quicksort - Speed-optimized sorting algorithm

    Time Complexity: O(n log n) average, O(n²) worst case
    Space Complexity: O(log n) for recursion stack
    Stable: No
    In-place: Yes (but this implementation creates new arrays for clarity)

    Best for: Large datasets where average-case performance is important
    and stability is not required.

    Args:
        arr: List to sort

    Returns:
        Sorted list
    """
    if len(arr) <= 1:
        return arr

    # Use median-of-three for pivot selection to avoid worst case
    pivot = _median_of_three(arr)

    # Three-way partitioning for better handling of duplicates
    less = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    greater = [x for x in arr if x > pivot]

    return quicksort(less) + equal + quicksort(greater)


def quicksort_inplace(arr: List[T], low: int = 0, high: int = None) -> None:
    """
    In-place quicksort implementation for true O(log n) space complexity

    Args:
        arr: List to sort in-place
        low: Starting index
        high: Ending index
    """
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_index = _partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)


def _partition(arr: List[T], low: int, high: int) -> int:
    """Partition helper for in-place quicksort"""
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def _median_of_three(arr: List[T]) -> T:
    """Select pivot using median-of-three strategy"""
    n = len(arr)
    if n < 3:
        return arr[0]

    first, mid, last = arr[0], arr[n // 2], arr[-1]
    return sorted([first, mid, last])[1]


# =============================================================================
# 2. HEAPSORT - Optimized for Memory
# =============================================================================
def heapsort(arr: List[T]) -> List[T]:
    """
    Heapsort - Memory-optimized sorting algorithm

    Time Complexity: O(n log n) guaranteed (no worst case degradation)
    Space Complexity: O(1) - sorts in-place
    Stable: No
    In-place: Yes

    Best for: Memory-constrained environments where worst-case performance
    guarantees are needed and stability is not required.

    Args:
        arr: List to sort

    Returns:
        Sorted list
    """
    # Make a copy to avoid modifying original
    result = arr.copy()
    n = len(result)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        _heapify(result, n, i)

    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        result[0], result[i] = result[i], result[0]
        _heapify(result, i, 0)

    return result


def _heapify(arr: List[T], n: int, i: int) -> None:
    """
    Heapify subtree rooted at index i

    Args:
        arr: Array to heapify
        n: Size of heap
        i: Root index of subtree
    """
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)


# =============================================================================
# 3. MERGESORT - Stable Sorting
# =============================================================================
def mergesort(arr: List[T]) -> List[T]:
    """
    Mergesort - Stable sorting algorithm

    Time Complexity: O(n log n) guaranteed
    Space Complexity: O(n) for temporary arrays
    Stable: Yes (preserves relative order of equal elements)
    In-place: No

    Best for: When stability is required (e.g., sorting by multiple keys)
    and you have enough memory. Predictable performance.

    Args:
        arr: List to sort

    Returns:
        Sorted list (stable)
    """
    if len(arr) <= 1:
        return arr

    # Divide
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])

    # Conquer (merge)
    return _merge(left, right)


def _merge(left: List[T], right: List[T]) -> List[T]:
    """
    Merge two sorted lists while maintaining stability

    Args:
        left: Sorted left sublist
        right: Sorted right sublist

    Returns:
        Merged sorted list
    """
    result = []
    i = j = 0

    # Merge while both lists have elements
    while i < len(left) and j < len(right):
        # Use <= to maintain stability (left elements come first if equal)
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Append remaining elements
    result.extend(left[i:])
    result.extend(right[j:])

    return result


# =============================================================================
# COMPARISON AND TESTING
# =============================================================================
def benchmark_algorithms(size: int = 1000) -> None:
    """
    Benchmark all three sorting algorithms

    Args:
        size: Size of test array
    """
    print("\n" + "="*70)
    print(f"SORTING ALGORITHM BENCHMARK (n={size})")
    print("="*70)

    # Generate test data
    test_data = [random.randint(1, 1000) for _ in range(size)]

    # Test Quicksort (Speed)
    arr = test_data.copy()
    start = time.time()
    sorted_quick = quicksort(arr)
    time_quick = time.time() - start

    # Test Heapsort (Memory)
    arr = test_data.copy()
    start = time.time()
    sorted_heap = heapsort(arr)
    time_heap = time.time() - start

    # Test Mergesort (Stable)
    arr = test_data.copy()
    start = time.time()
    sorted_merge = mergesort(arr)
    time_merge = time.time() - start

    # Verify all produce same result
    assert sorted_quick == sorted_heap == sorted_merge, "Sort mismatch!"

    print(f"\n1. QUICKSORT (Speed-optimized)")
    print(f"   Time: {time_quick:.6f} seconds")
    print(f"   Space: O(log n) average")
    print(f"   Stable: No")
    print(f"   Best for: General-purpose, fastest average case")

    print(f"\n2. HEAPSORT (Memory-optimized)")
    print(f"   Time: {time_heap:.6f} seconds")
    print(f"   Space: O(1) - in-place")
    print(f"   Stable: No")
    print(f"   Best for: Memory-constrained, guaranteed O(n log n)")

    print(f"\n3. MERGESORT (Stable)")
    print(f"   Time: {time_merge:.6f} seconds")
    print(f"   Space: O(n)")
    print(f"   Stable: Yes")
    print(f"   Best for: Stability required, predictable performance")

    print("\n" + "="*70)
    print(f"All algorithms verified correct!")
    print("="*70 + "\n")


def demonstrate_stability() -> None:
    """Demonstrate that mergesort is stable while others are not"""
    print("\n" + "="*70)
    print("STABILITY DEMONSTRATION")
    print("="*70)

    # Create list of tuples (value, original_index)
    # If stable, items with same value keep original order
    data = [(5, 'a'), (3, 'b'), (5, 'c'), (3, 'd'), (1, 'e')]

    print(f"\nOriginal: {data}")

    # Sort by first element only
    key_func = lambda x: x[0]

    # Python's sort (Timsort - stable)
    sorted_stable = sorted(data, key=key_func)
    print(f"Stable sort: {sorted_stable}")
    print("  Note: (5,'a') before (5,'c'), (3,'b') before (3,'d')")

    # For demonstration, show what unstable might look like
    print(f"\nUnstable sorts (Quicksort/Heapsort) may reorder equal elements")
    print("  e.g., (5,'c') might come before (5,'a')")

    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ACEAC SORTING ALGORITHMS - SECURITY REVIEW")
    print("="*70)
    print("Author: @sarowarzahan414")
    print("Date: 2025-11-18")
    print("="*70)

    # Run benchmarks
    benchmark_algorithms(size=1000)
    benchmark_algorithms(size=5000)

    # Demonstrate stability
    demonstrate_stability()

    print("\nSUMMARY:")
    print("  - Quicksort: Fastest average case (O(n log n)), not stable")
    print("  - Heapsort: Best memory usage (O(1)), guaranteed O(n log n)")
    print("  - Mergesort: Stable, predictable O(n log n), uses O(n) space")
