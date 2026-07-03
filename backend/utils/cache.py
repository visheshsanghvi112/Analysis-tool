import time
from functools import wraps
from threading import Lock
import copy

class TTLInMemoryCache:
    """
    A thread-safe In-Memory Cache with TTL (Time To Live).
    Useful for caching expensive API responses and analytical calculations.
    """
    def __init__(self):
        self._cache = {}
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if time.time() < expires_at:
                    # Return a deep copy to prevent mutation of cached data
                    return copy.deepcopy(value)
                del self._cache[key]
        return None

    def set(self, key, value, ttl_seconds):
        with self._lock:
            # Store a deep copy of the value to prevent external mutations from affecting cache
            self._cache[key] = (copy.deepcopy(value), time.time() + ttl_seconds)

    def clear(self):
        with self._lock:
            self._cache.clear()

# Global cache store instance
cache_store = TTLInMemoryCache()

def cache_ttl(seconds=300):
    """
    Decorator to cache a function's return value for a specified duration.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique cache key based on function name and args/kwargs
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            cached_val = cache_store.get(key)
            if cached_val is not None:
                return cached_val
            
            result = func(*args, **kwargs)
            # Only cache non-empty/successful results
            if result is not None:
                cache_store.set(key, result, seconds)
            return result
        return wrapper
    return decorator
