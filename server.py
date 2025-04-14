from flask import Flask, request, jsonify
import numpy as np
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import networkx as nx
import time
import requests
import random
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
from Persistent import PersistentPaperCache
import os
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv("MONGODB_URI")
paper_cache = PersistentPaperCache(max_size=10000, ttl=600, mongodb_uri=mongo_uri)


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Precomputed storage
tfidf_vectorizer = None
tfidf_matrix = None


def maintenance_thread():
    while True:
        time.sleep(300)  # Run every 5 minutes
        paper_cache.cleanup_expired()
        paper_cache.save_all()
        logger.info(f"Cache maintenance completed. Current stats: {paper_cache.stats}")


# Start the maintenance thread
maintenance_thread = threading.Thread(target=maintenance_thread, daemon=True)
maintenance_thread.start()


def fetch_with_retry(url, max_retries=5, backoff_factor=2, timeout=10):
    """
    Fetch data from URL with exponential backoff retry mechanism.
    Handles various types of errors and implements proper backoff.
    """
    retries = 0
    while retries < max_retries:
        try:
            # Attempt the request with timeout
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Will raise an error for 4xx/5xx status codes
            return response.json()  # Return JSON response

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                # Too many requests, apply exponential backoff
                retries += 1
                wait_time = backoff_factor**retries + random.uniform(0, 1)
                logger.warning(
                    f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds..."
                )
                time.sleep(wait_time)  # Wait before retrying
            elif 500 <= response.status_code < 600:
                # Server errors might be temporary, retry
                retries += 1
                wait_time = backoff_factor**retries + random.uniform(0, 1)
                logger.warning(
                    f"Server error {response.status_code}. Retrying in {wait_time:.2f} seconds..."
                )
                time.sleep(wait_time)
            else:
                # For other HTTP errors, just raise them
                logger.error(f"HTTP error {response.status_code}: {e}")
                raise

        except requests.exceptions.Timeout:
            # Handle timeouts
            retries += 1
            wait_time = backoff_factor**retries + random.uniform(0, 1)
            logger.warning(f"Request timed out. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

        except requests.exceptions.ConnectionError:
            # Handle connection errors
            retries += 1
            wait_time = backoff_factor**retries + random.uniform(0, 1)
            logger.warning(f"Connection error. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

        except Exception as e:
            # General exception handling
            logger.error(f"Unexpected error during request: {str(e)}")
            raise

    raise Exception(f"Max retries ({max_retries}) reached for URL: {url}")


def transform_text_spacy(text):
    """Preprocess text using SpaCy for tokenization and lemmatization."""
    if not isinstance(text, str) or not text:
        return ""

    text = text.lower()
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha
        and token.text not in STOP_WORDS
        and token.text not in string.punctuation
    ]

    return " ".join(tokens)


def search_papers_openalex(query, top_n=5):
    """Search papers on OpenAlex API with retry mechanism."""
    url = f"https://api.openalex.org/works?search={query}&per-page={top_n}"
    try:
        data = fetch_with_retry(url)
        results = []
        for item in data["results"]:
            paper_id = item["id"].split("/")[-1]  # Extract OpenAlex ID
            paper_data = {
                "paperId": paper_id,
                "title": item.get("title", ""),
                "abstract": item.get("abstract_inverted_index", ""),
                "year": item.get("publication_year", "Unknown"),
                "citationCount": item.get("cited_by_count", 0),
                "url": data.get("id", f"https://openalex.org/{paper_id}"),
            }

            paper_cache.set(paper_id, paper_data)
            results.append(paper_data)

        logger.info(f"Found {len(results)} papers matching query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error fetching OpenAlex results: {e}")
        return []


def get_paper_details_openalex(paper_id):
    """
    Fetch paper details from OpenAlex with proper caching.
    """
    # Check if paper is already in cache
    cached_paper = paper_cache.get(paper_id)
    if cached_paper:
        logger.debug(f"Cache hit for paper {paper_id}")
        return cached_paper

    logger.debug(f"Cache miss for paper {paper_id}, fetching from API")
    url = f"https://api.openalex.org/works/https://openalex.org/{paper_id}"

    try:
        data = fetch_with_retry(url)

        abstract = ""
        if data.get("abstract_inverted_index"):
            try:
                # Reconstruct full abstract from inverted index
                inverted_index = data["abstract_inverted_index"]
                index_map = {}
                for word, positions in inverted_index.items():
                    for pos in positions:
                        index_map[pos] = word

                # Ensure all positions are covered
                max_position = max(index_map.keys()) if index_map else -1
                words = []
                for i in range(max_position + 1):
                    words.append(index_map.get(i, ""))

                abstract = " ".join(words).strip()

                # Fallback if reconstruction failed
                if not abstract or abstract.isspace():
                    abstract = "Abstract reconstruction failed"
            except Exception as e:
                logger.error(f"Error reconstructing abstract for {paper_id}: {e}")
                abstract = "Error processing abstract"
        else:
            abstract = "Abstract not available"

        paper_details = {
            "paperId": paper_id,
            "title": data.get("title", "Unknown Title"),
            "abstract": abstract,
            "year": data.get("publication_year", "Year not available"),
            "citationCount": data.get("cited_by_count", 0),
            "url": data.get(
                "id", f"https://openalex.org/{paper_id}"
            ),  # Fallback if missing
        }

        # Store in cache
        paper_cache.set(paper_id, paper_details)
        return paper_details

    except Exception as e:
        logger.error(f"Error fetching details for {paper_id}: {e}")
        fallback_details = {
            "paperId": paper_id,
            "title": "Unknown Title",
            "abstract": "Abstract not available",
            "year": "Year not available",
            "citationCount": 0,
            "url": f"https://openalex.org/{paper_id}",
        }
        # Don't cache errors permanently, but for a shorter time
        paper_cache.set(paper_id, fallback_details)
        return fallback_details


def get_rwr_recommendations(
    G, seed_paper, restart_prob=0.15, max_iter=100, tol=1e-6, min_score_threshold=0.01
):
    """Performs Random Walk with Restart (RWR) on the graph for a given seed paper."""

    if not G.has_node(seed_paper):
        raise ValueError(f"Seed paper {seed_paper} not found in graph.")

    # Get subgraph of the connected component
    component = nx.node_connected_component(G, seed_paper)
    G_sub = G.subgraph(component).copy()
    nodes = list(G_sub.nodes())
    num_nodes = len(nodes)

    # Index mapping
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Create normalized adjacency matrix
    adj_matrix = nx.to_numpy_array(G_sub, nodelist=nodes, weight="weight")
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = adj_matrix / row_sums

    # Initialize probability distribution
    p = np.zeros(num_nodes)
    p[node_idx[seed_paper]] = 1.0

    # Perform power iteration
    for _ in range(max_iter):
        new_p = (1 - restart_prob) * np.dot(transition_matrix.T, p) + restart_prob * p
        if np.linalg.norm(new_p - p, 1) < tol:
            break
        p = new_p

    # Convert results to dictionary
    scores = {nodes[i]: p[i] for i in range(num_nodes)}

    # Filter by threshold
    filtered_recommendations = {
        paper: score for paper, score in scores.items() if score > min_score_threshold
    }

    # Sort recommendations by score
    sorted_recommendations = dict(
        sorted(filtered_recommendations.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_recommendations


def fetch_neighbors_openalex(paper_id, sample_size=5):
    """Fetch up to `sample_size` citations and references for a paper."""
    cited_ids = []
    citing_ids = []

    # References (papers this paper cites)
    url = f"https://api.openalex.org/works/https://openalex.org/{paper_id}"
    try:
        data = fetch_with_retry(url)
        referenced = data.get("referenced_works", [])
        cited_ids = [ref.split("/")[-1] for ref in referenced]
    except Exception as e:
        logger.error(f"Error fetching references for {paper_id}: {e}")

    # Citations (papers that cite this paper)
    url_citers = f"https://api.openalex.org/works?filter=cites:W{paper_id}&per-page=20"
    try:
        data = fetch_with_retry(url_citers)
        citing_ids = [item["id"].split("/")[-1] for item in data.get("results", [])]
    except Exception as e:
        logger.error(f"Error fetching citations for {paper_id}: {e}")

    # Sample a subset to avoid overload
    sampled_cited = random.sample(cited_ids, min(len(cited_ids), sample_size))
    sampled_citing = random.sample(citing_ids, min(len(citing_ids), sample_size))

    return sampled_cited + sampled_citing


def batch_fetch_details(paper_ids):
    """Fetch paper details in batches with improved thread management."""
    # Create a set to remove duplicates
    unique_ids = set(paper_ids)

    # First check cache for already fetched papers
    to_fetch = []
    results = []

    for pid in unique_ids:
        if paper_cache.get(pid) is not None:
            logger.info(f"Paper {pid} found in cache")
            results.append(paper_cache.get(pid))
        else:
            to_fetch.append(pid)

    # Only fetch papers not in cache
    if to_fetch:
        logger.info(f"Batch fetching {len(to_fetch)} papers not in cache")
        with ThreadPoolExecutor(max_workers=10) as executor:
            new_results = list(executor.map(get_paper_details_openalex, to_fetch))
            results.extend(new_results)

    return results


def batch_add_recommendations(paper_ids):
    """Add recommended papers in batches with cache utilization."""
    # Use the improved batch_fetch_details which checks cache first
    return batch_fetch_details(paper_ids)


def generate_recommendations_dynamic(query, top_n=5):
    logger.info(f"Processing query: {query}")

    seed_results = search_papers_openalex(query, top_n)
    logger.info(f"Found {len(seed_results)} seed papers from OpenAlex.")

    G = nx.Graph()

    for seed in seed_results:
        seed_id = seed["paperId"]
        G.add_node(seed_id, **seed)
        logger.info(f"Added seed paper: {seed_id} — {seed['title']}")

        neighbors = fetch_neighbors_openalex(seed_id)
        logger.info(
            f"{len(neighbors)} total neighbors (citations + references) found for {seed_id}."
        )

        details_list = batch_fetch_details(neighbors)
        for neighbor_details in details_list:
            neighbor_id = neighbor_details["paperId"]
            G.add_node(neighbor_id, **neighbor_details)
            G.add_edge(seed_id, neighbor_id, weight=1.0)
            logger.debug(f"Added neighbor paper: {neighbor_id}")

    logger.info(
        f"Graph constructed with {len(G.nodes)} nodes and {len(G.edges)} edges."
    )

    # Compute abstract similarity weights
    edge_count = 0
    for u, v in G.edges():
        abs_u = G.nodes[u].get("abstract", "")
        abs_v = G.nodes[v].get("abstract", "")
        if isinstance(abs_u, str) and isinstance(abs_v, str):
            try:
                vec = tfidf_vectorizer.transform(
                    [transform_text_spacy(abs_u), transform_text_spacy(abs_v)]
                )
                sim = cosine_similarity(vec[0], vec[1])[0][0]
                G[u][v]["weight"] = sim
                edge_count += 1
            except Exception as e:
                logger.warning(f"Error computing similarity between {u} and {v}: {e}")
    logger.info(f"Computed similarity weights for {edge_count} edges.")

    # RWR
    recommendations = {}
    for seed in seed_results:
        try:
            logger.info(f"Running RWR for seed paper {seed['paperId']}")
            rwr_scores = get_rwr_recommendations(G, seed["paperId"])
            top_rwr = list(rwr_scores.keys())[:10]
            logger.info(
                f"Top {len(top_rwr)} papers returned from RWR for {seed['paperId']}."
            )

            # Add recommendations in parallel
            recommendations_papers = batch_add_recommendations(top_rwr)
            for paper in recommendations_papers:
                pid = paper["paperId"]
                if pid not in recommendations:
                    recommendations[pid] = paper
                    logger.debug(f"Added recommendation: {pid}")
        except Exception as e:
            logger.error(f"Error running RWR for {seed['paperId']}: {e}")

    logger.info(f"Total recommendations collected: {len(recommendations)}")
    return list(recommendations.values())


def add_cache_routes(app):
    @app.route("/api/cache/stats", methods=["GET"])
    def get_cache_stats():
        """API endpoint to get cache statistics."""
        return jsonify(paper_cache.stats)

    @app.route("/api/cache/items", methods=["GET"])
    def get_cache_items():
        """API endpoint to view cached items."""
        limit = request.args.get("limit", 100, type=int)
        include_data = request.args.get("include_data", "false").lower() == "true"
        items = paper_cache.get_all_items(limit=limit, include_data=include_data)
        return jsonify(
            {
                "total_items": paper_cache.size,
                "items_returned": len(items),
                "items": items,
            }
        )

    @app.route("/api/cache/clear", methods=["POST"])
    def clear_cache():
        """API endpoint to clear the cache."""
        paper_cache.clear()
        return jsonify({"status": "success", "message": "Cache cleared"})

    @app.route("/api/cache/save", methods=["POST"])
    def save_cache():
        """API endpoint to force save the cache."""
        paper_cache.save_all()
        return jsonify({"status": "success", "message": "Cache saved to disk"})

    @app.route("/api/cache/item/<paper_id>", methods=["DELETE"])
    def remove_cache_item(paper_id):
        """API endpoint to remove a specific item from cache."""
        paper_cache.remove(paper_id)
        return jsonify(
            {"status": "success", "message": f"Item {paper_id} removed from cache"}
        )

    logger.info("Cache management endpoints registered")


@app.route("/api/search", methods=["POST"])
def search():
    """API endpoint for searching research papers."""
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    results = generate_recommendations_dynamic(query)
    return jsonify({"query": query, "results": results})


if __name__ == "__main__":
    app.run(debug=True)
