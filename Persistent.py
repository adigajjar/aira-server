import threading
import time
import logging
from collections import OrderedDict
from datetime import datetime
from pymongo import MongoClient
import pickle
import base64
import os

# Configure logger
logger = logging.getLogger(__name__)


class PersistentPaperCache:
    """
    Thread-safe persistent paper cache with TTL and max size limit.
    Uses LRU eviction policy and persists to MongoDB.
    """

    def __init__(
        self,
        max_size=1000,
        ttl=7200,
        mongodb_uri=None,
        db_name="paper_cache_db",
        collection_name="paper_cache",
    ):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of items in cache
            ttl: Time to live for cache items in seconds (default 2 hours)
            mongodb_uri: MongoDB connection URI (defaults to environment variable MONGODB_URI)
            db_name: MongoDB database name
            collection_name: MongoDB collection name
        """
        self._cache = OrderedDict()  # For LRU implementation
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._hit_count = 0
        self._miss_count = 0

        # MongoDB connection
        self._mongodb_uri = mongodb_uri or os.environ.get("MONGODB_URI")
        if not self._mongodb_uri:
            raise ValueError(
                "MongoDB URI must be provided either as parameter or as MONGODB_URI environment variable"
            )

        self._db_name = db_name
        self._collection_name = collection_name
        self._client = MongoClient(self._mongodb_uri)
        self._db = self._client[self._db_name]
        self._collection = self._db[self._collection_name]

        # Create TTL index to automatically expire documents
        self._collection.create_index("timestamp", expireAfterSeconds=self._ttl)

        # Load cache from MongoDB
        self._load_cache()

        # Set up periodic saving
        self._last_save_time = time.time()
        self._save_interval = 300  # Save metadata every 5 minutes

    def _load_cache(self):
        """Load cache from MongoDB."""
        try:
            # Load stats
            stats_doc = self._db["cache_stats"].find_one({"_id": "stats"})
            if stats_doc:
                self._hit_count = stats_doc.get("hits", 0)
                self._miss_count = stats_doc.get("misses", 0)

            # Load only non-expired items
            now = time.time()
            cursor = self._collection.find({"timestamp": {"$gt": now - self._ttl}})

            for doc in cursor:
                key = doc["_id"]
                value = pickle.loads(base64.b64decode(doc["value"]))
                timestamp = doc["timestamp"]
                self._cache[key] = (value, timestamp)

            logger.info(f"Loaded {len(self._cache)} items from MongoDB cache")

        except Exception as e:
            logger.error(f"Error loading cache from MongoDB: {e}")
            # Start with empty cache if loading fails
            self._cache = OrderedDict()

    def _save_metadata(self):
        """Save cache metadata to MongoDB."""
        try:
            # Save stats
            self._db["cache_stats"].update_one(
                {"_id": "stats"},
                {
                    "$set": {
                        "hits": self._hit_count,
                        "misses": self._miss_count,
                        "last_save": time.time(),
                        "size": len(self._cache),
                        "max_size": self._max_size,
                        "ttl": self._ttl,
                    }
                },
                upsert=True,
            )

        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")

    def _save_item(self, key, value, timestamp):
        """Save a single cache item to MongoDB."""
        try:
            # Convert value to binary and then to base64 for storage
            value_binary = base64.b64encode(pickle.dumps(value)).decode("utf-8")

            self._collection.update_one(
                {"_id": key},
                {
                    "$set": {
                        "value": value_binary,
                        "timestamp": timestamp,
                        "created": datetime.fromtimestamp(timestamp),
                        "size": len(value_binary),
                    }
                },
                upsert=True,
            )
        except Exception as e:
            logger.error(f"Error saving cache item {key}: {e}")

    def _save_if_needed(self):
        """Save cache metadata to MongoDB if enough time has passed."""
        now = time.time()
        if now - self._last_save_time > self._save_interval:
            self._save_metadata()
            self._last_save_time = now

    def get(self, key):
        """
        Get an item from the cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found or expired
        """
        with self._lock:
            if key not in self._cache:
                # Check MongoDB directly in case it was added by another instance
                doc = self._collection.find_one({"_id": key})
                if doc and time.time() - doc["timestamp"] <= self._ttl:
                    # Add to in-memory cache
                    value = pickle.loads(base64.b64decode(doc["value"]))
                    timestamp = doc["timestamp"]
                    self._cache[key] = (value, timestamp)
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hit_count += 1

                    # Periodically save cache metadata
                    self._save_if_needed()

                    return value

                self._miss_count += 1
                return None

            item, timestamp = self._cache[key]

            # Check if item is expired
            if time.time() - timestamp > self._ttl:
                self._remove_item(key)
                self._miss_count += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hit_count += 1

            # Periodically save cache metadata
            self._save_if_needed()

            return item

    def set(self, key, value):
        """
        Add or update an item in the cache.

        Args:
            key: Cache key
            value: Item to cache
        """
        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._remove_item(key)

            # If cache is full, remove oldest item (LRU eviction)
            if len(self._cache) >= self._max_size:
                oldest_key, _ = self._cache.popitem(
                    last=False
                )  # Remove from the beginning (oldest)
                self._remove_item_from_mongodb(oldest_key)

            # Add new item
            timestamp = time.time()
            self._cache[key] = (value, timestamp)

            # Save the item to MongoDB
            self._save_item(key, value, timestamp)

            # Periodically save cache metadata
            self._save_if_needed()

    def _remove_item(self, key):
        """Remove an item from cache and MongoDB."""
        if key in self._cache:
            self._cache.pop(key)
            self._remove_item_from_mongodb(key)

    def _remove_item_from_mongodb(self, key):
        """Remove a cache item from MongoDB."""
        try:
            self._collection.delete_one({"_id": key})
        except Exception as e:
            logger.error(f"Error removing cache item for {key} from MongoDB: {e}")

    def remove(self, key):
        """Remove an item from the cache."""
        with self._lock:
            self._remove_item(key)

    def clear(self):
        """Clear the entire cache."""
        with self._lock:
            # Clear in-memory cache
            self._cache.clear()

            # Clear MongoDB cache
            try:
                self._collection.delete_many({})
            except Exception as e:
                logger.error(f"Error clearing MongoDB cache: {e}")

            # Save empty metadata
            self._save_metadata()

    def cleanup_expired(self):
        """Remove all expired items from the cache."""
        now = time.time()
        with self._lock:
            expired_keys = [
                key
                for key, (_, timestamp) in self._cache.items()
                if now - timestamp > self._ttl
            ]
            for key in expired_keys:
                self._remove_item(key)

            # MongoDB TTL index should handle expiration automatically,
            # but we can force a cleanup for consistency
            try:
                self._collection.delete_many({"timestamp": {"$lt": now - self._ttl}})
            except Exception as e:
                logger.error(f"Error cleaning up expired items from MongoDB: {e}")

            if expired_keys:
                # Save metadata after cleanup
                self._save_metadata()
                logger.info(f"Cleaned up {len(expired_keys)} expired items")

    def get_all_items(self, limit=100, include_data=False):
        """
        Get all items in the cache (for viewing purposes).

        Args:
            limit: Maximum number of items to return
            include_data: Whether to include the actual cached data

        Returns:
            Dictionary of cache items
        """
        result = {}
        with self._lock:
            count = 0
            now = time.time()
            # Get the most recent items from MongoDB
            cursor = self._collection.find().sort("timestamp", -1).limit(limit)

            for doc in cursor:
                key = doc["_id"]
                timestamp = doc["timestamp"]
                age = now - timestamp
                is_expired = age > self._ttl

                item_data = {
                    "timestamp": timestamp,
                    "age_seconds": int(age),
                    "created": datetime.fromtimestamp(timestamp).isoformat(),
                    "expired": is_expired,
                }

                if include_data:
                    # Load the value from MongoDB if not in memory cache
                    if key in self._cache:
                        value = self._cache[key][0]
                    else:
                        value = pickle.loads(base64.b64decode(doc["value"]))
                    item_data["data"] = value

                result[key] = item_data
                count += 1
                if count >= limit:
                    break

        return result

    def save_all(self):
        """Force save all cache items and metadata."""
        with self._lock:
            # Save all items
            for key, (value, timestamp) in self._cache.items():
                self._save_item(key, value, timestamp)

            # Save metadata
            self._save_metadata()
            logger.info(f"Saved {len(self._cache)} items to MongoDB cache")

    @property
    def size(self):
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    @property
    def stats(self):
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0

            # Calculate MongoDB usage
            db_size = 0
            try:
                db_stats = self._db.command("collStats", self._collection_name)
                db_size = db_stats.get("size", 0)
            except Exception:
                db_size = -1

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl": self._ttl,
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_rate": f"{hit_rate:.2%}",
                "db_size_bytes": db_size,
                "db_size_mb": (
                    f"{db_size / (1024 * 1024):.2f} MB" if db_size > 0 else "Unknown"
                ),
                "mongodb_db": self._db_name,
                "mongodb_collection": self._collection_name,
            }

    def __del__(self):
        """Close MongoDB connection when object is destroyed."""
        if hasattr(self, "_client"):
            self._client.close()
