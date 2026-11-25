"""Seed functions for the bug-fixing tasks: correct code + tests.

Each seed has visible tests (their failures are shown to the model) and
hidden tests (a superset, with edge cases, run only to compute the reward).
Hidden tests are what stop a model from passing by special-casing the inputs
it was shown.
"""

SEEDS = [
    dict(name="running_max", code='''
def running_max(xs):
    """Return a list where element i is the largest value in xs[0..i]."""
    out = []
    best = None
    for x in xs:
        if best is None or x > best:
            best = x
        out.append(best)
    return out
''', visible=["running_max([1, 3, 2, 5, 4]) == [1, 3, 3, 5, 5]", "running_max([]) == []"],
        hidden=["running_max([-5, -2, -9]) == [-5, -2, -2]", "running_max([7]) == [7]", "running_max([2, 2, 1]) == [2, 2, 2]"]),
    dict(name="count_vowels", code='''
def count_vowels(s):
    """Count the vowels (a, e, i, o, u, either case) in s."""
    n = 0
    for ch in s.lower():
        if ch in "aeiou":
            n += 1
    return n
''', visible=['count_vowels("Hello World") == 3', 'count_vowels("") == 0'],
        hidden=['count_vowels("AEIOU") == 5', 'count_vowels("rhythm") == 0', 'count_vowels("Programming") == 3']),
    dict(name="fizzbuzz", code='''
def fizzbuzz(n):
    """Return the FizzBuzz strings for 1..n inclusive."""
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out
''', visible=['fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"]', "fizzbuzz(0) == []"],
        hidden=['fizzbuzz(15)[-1] == "FizzBuzz"', "len(fizzbuzz(100)) == 100", 'fizzbuzz(10)[8] == "Fizz"', 'fizzbuzz(1) == ["1"]']),
    dict(name="is_palindrome", code='''
def is_palindrome(s):
    """True if s reads the same forwards and backwards, ignoring case and non-alphanumeric characters."""
    chars = [c.lower() for c in s if c.isalnum()]
    i, j = 0, len(chars) - 1
    while i < j:
        if chars[i] != chars[j]:
            return False
        i += 1
        j -= 1
    return True
''', visible=['is_palindrome("A man, a plan, a canal: Panama") == True', 'is_palindrome("hello") == False'],
        hidden=['is_palindrome("") == True', 'is_palindrome("ab") == False', "is_palindrome(\"No 'x' in Nixon\") == True", 'is_palindrome("abca") == False']),
    dict(name="binary_search", code='''
def binary_search(xs, target):
    """Index of target in the sorted list xs, or -1 if absent."""
    lo, hi = 0, len(xs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''', visible=["binary_search([1, 3, 5, 7, 9], 7) == 3", "binary_search([1, 3, 5], 4) == -1", "binary_search([2], 2) == 0"],
        hidden=["binary_search([], 1) == -1", "binary_search([1, 3, 5, 7, 9], 1) == 0", "binary_search([1, 3, 5, 7, 9], 9) == 4", "binary_search([1, 3, 5, 7, 9], 10) == -1"]),
    dict(name="merge_sorted", code='''
def merge_sorted(a, b):
    """Merge two sorted lists into one sorted list."""
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
''', visible=["merge_sorted([1, 4, 6], [2, 3, 7]) == [1, 2, 3, 4, 6, 7]", "merge_sorted([], [1]) == [1]"],
        hidden=["merge_sorted([1, 1], [1]) == [1, 1, 1]", "merge_sorted([5], []) == [5]", "merge_sorted([1, 2, 3], [4, 5, 6]) == [1, 2, 3, 4, 5, 6]", "merge_sorted([4, 5], [1, 2]) == [1, 2, 4, 5]"]),
    dict(name="chunk", code='''
def chunk(xs, n):
    """Split xs into consecutive lists of length n (the last one may be shorter)."""
    out = []
    for i in range(0, len(xs), n):
        out.append(xs[i:i + n])
    return out
''', visible=["chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "chunk([], 3) == []"],
        hidden=["chunk([1, 2, 3], 3) == [[1, 2, 3]]", "chunk([1, 2, 3], 1) == [[1], [2], [3]]", "chunk([1, 2, 3, 4], 5) == [[1, 2, 3, 4]]"]),
    dict(name="second_largest", code='''
def second_largest(xs):
    """The second-largest distinct value in xs, or None if there isn't one."""
    first = second = None
    for x in xs:
        if first is None or x > first:
            second = first
            first = x
        elif x < first and (second is None or x > second):
            second = x
    return second
''', visible=["second_largest([3, 1, 4, 1, 5]) == 4", "second_largest([7]) == None"],
        hidden=["second_largest([2, 2, 2]) == None", "second_largest([5, 5, 3]) == 3", "second_largest([-1, -2]) == -2", "second_largest([1, 9, 9, 8]) == 8"]),
    dict(name="rotate", code='''
def rotate(xs, k):
    """Rotate xs right by k positions (k may exceed len(xs))."""
    if not xs:
        return []
    k = k % len(xs)
    if k == 0:
        return list(xs)
    return xs[-k:] + xs[:-k]
''', visible=["rotate([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]", "rotate([], 3) == []"],
        hidden=["rotate([1, 2, 3], 3) == [1, 2, 3]", "rotate([1, 2, 3], 4) == [3, 1, 2]", "rotate([1], 10) == [1]", "rotate([1, 2], 0) == [1, 2]"]),
    dict(name="moving_average", code='''
def moving_average(xs, k):
    """Averages of each window of k consecutive values."""
    out = []
    total = sum(xs[:k])
    if len(xs) >= k:
        out.append(total / k)
    for i in range(k, len(xs)):
        total += xs[i] - xs[i - k]
        out.append(total / k)
    return out
''', visible=["moving_average([1, 2, 3, 4, 5], 2) == [1.5, 2.5, 3.5, 4.5]", "moving_average([1, 2], 3) == []"],
        hidden=["moving_average([4, 4, 4], 3) == [4.0]", "moving_average([1, 2, 3, 4], 1) == [1.0, 2.0, 3.0, 4.0]", "moving_average([10, 0, 10, 0], 2) == [5.0, 5.0, 5.0]"]),
    dict(name="gcd", code='''
def gcd(a, b):
    """Greatest common divisor of two non-negative integers."""
    while b:
        a, b = b, a % b
    return a
''', visible=["gcd(12, 18) == 6", "gcd(7, 5) == 1"],
        hidden=["gcd(0, 5) == 5", "gcd(5, 0) == 5", "gcd(100, 75) == 25", "gcd(17, 17) == 17"]),
    dict(name="is_prime", code='''
def is_prime(n):
    """True if n is a prime number."""
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
''', visible=["is_prime(7) == True", "is_prime(9) == False"],
        hidden=["is_prime(1) == False", "is_prime(2) == True", "is_prime(25) == False", "is_prime(97) == True", "is_prime(0) == False"]),
    dict(name="digit_sum", code='''
def digit_sum(n):
    """Sum of the decimal digits of a non-negative integer."""
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total
''', visible=["digit_sum(1234) == 10", "digit_sum(0) == 0"],
        hidden=["digit_sum(9) == 9", "digit_sum(1000) == 1", "digit_sum(9999) == 36"]),
    dict(name="compress", code='''
def compress(s):
    """Run-length encode s: "aaabcc" -> "a3b1c2"."""
    if not s:
        return ""
    out = []
    prev, count = s[0], 1
    for ch in s[1:]:
        if ch == prev:
            count += 1
        else:
            out.append(f"{prev}{count}")
            prev, count = ch, 1
    out.append(f"{prev}{count}")
    return "".join(out)
''', visible=['compress("aaabcc") == "a3b1c2"', 'compress("") == ""'],
        hidden=['compress("a") == "a1"', 'compress("abab") == "a1b1a1b1"', 'compress("zzzz") == "z4"']),
    dict(name="median", code='''
def median(xs):
    """Median of a non-empty list of numbers."""
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
''', visible=["median([3, 1, 2]) == 2", "median([4, 1, 3, 2]) == 2.5"],
        hidden=["median([5]) == 5", "median([1, 2]) == 1.5", "median([7, 7, 7, 1]) == 7.0", "median([10, 2, 8, 4, 6]) == 6"]),
    dict(name="transpose", code='''
def transpose(m):
    """Transpose a rectangular matrix given as a list of rows."""
    if not m:
        return []
    return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]
''', visible=["transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]", "transpose([]) == []"],
        hidden=["transpose([[1]]) == [[1]]", "transpose([[1, 2]]) == [[1], [2]]", "transpose([[1], [2], [3]]) == [[1, 2, 3]]"]),
    dict(name="dedupe", code='''
def dedupe(xs):
    """Remove duplicates from xs, keeping the first occurrence of each value in order."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
''', visible=["dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]", "dedupe([]) == []"],
        hidden=["dedupe([1, 1, 1]) == [1]", 'dedupe(["b", "a", "b"]) == ["b", "a"]', "dedupe([1, 2, 3]) == [1, 2, 3]"]),
    dict(name="word_count", code='''
def word_count(text):
    """Count how many times each lowercase word appears; words are split on whitespace."""
    counts = {}
    for w in text.lower().split():
        counts[w] = counts.get(w, 0) + 1
    return counts
''', visible=['word_count("the cat the hat") == {"the": 2, "cat": 1, "hat": 1}', 'word_count("") == {}'],
        hidden=['word_count("A a A") == {"a": 3}', 'word_count("one") == {"one": 1}', 'word_count("x  y\\nx") == {"x": 2, "y": 1}']),
    dict(name="clamp", code='''
def clamp(x, lo, hi):
    """Limit x to the range [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
''', visible=["clamp(5, 0, 3) == 3", "clamp(-1, 0, 3) == 0"],
        hidden=["clamp(2, 0, 3) == 2", "clamp(0, 0, 3) == 0", "clamp(3, 0, 3) == 3", "clamp(10, 10, 10) == 10"]),
    dict(name="max_subarray", code='''
def max_subarray(xs):
    """Largest sum of a non-empty contiguous slice of xs."""
    best = cur = xs[0]
    for x in xs[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
''', visible=["max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6", "max_subarray([1]) == 1"],
        hidden=["max_subarray([-3, -1, -2]) == -1", "max_subarray([5, -1, 5]) == 9", "max_subarray([2, 3]) == 5"]),
]
