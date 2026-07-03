from slowapi import Limiter
from slowapi.util import get_remote_address

# Thread-safe rate limiter instance shared across all APIRouters
limiter = Limiter(key_func=get_remote_address)
