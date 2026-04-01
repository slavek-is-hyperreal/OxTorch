import sqlite3
import numpy as np
import oxtorch as torch
import struct
import argparse
import os
import time

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return 2147483647 * image_id1 + image_id2

class OxMatcher:
    def __init__(self, database_path, device='vulkan'):
        self.db = sqlite3.connect(database_path)
        self.device = device
        print(f"[OxMatcher] Initialized on {device}")

    def get_descriptor_by_id(self, image_id):
        """Fetches and normalizes a single descriptor set from SQLite."""
        cursor = self.db.cursor()
        cursor.execute("SELECT rows, cols, data FROM descriptors WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
        if row is None or row[2] is None:
            return None
        
        rows, cols, data = row
        desc = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols).astype(np.float32)
        # Normalize for cosine similarity (L2 norm)
        norm = np.linalg.norm(desc, axis=1, keepdims=True)
        desc /= (norm + 1e-7)
        return torch.from_numpy(desc)

    def match_pair(self, desc1, desc2, ratio_threshold=0.8):
        """Matches two descriptor sets using Vulkan MatMul and Ratio Test."""
        # Move to Vulkan for the heavy lifting
        t1 = desc1.to(self.device)
        t2 = desc2.to(self.device)
        
        # 1. Similarity Matrix (Dot Product since normalized)
        # S shape: (N1, N2)
        S = torch.matmul(t1, t2.transpose())
        
        # 2. Find best and second best matches (CPU fallback for argmax/topk if not native)
        # NOTE: Current oxtorch might need fallback for topk. 
        # We'll pull similarity matrix to RAM for the final selection logic 
        # to ensure 100% geometric parity with Lowe's Ratio Test.
        S_np = S.to_numpy()
        
        # [Debug] Analyze similarity values
        sim_max = S_np.max()
        sim_min = S_np.min()
        sim_avg = S_np.mean()
        print(f"    [Math-Debug] Similarity Matrix: min={sim_min:.4f}, max={sim_max:.4f}, avg={sim_avg:.4f}")
        
        # Cross-check: for each in A, find best in B
        best_idx = np.argmax(S_np, axis=1)
        best_vals = np.take_along_axis(S_np, best_idx[:, None], axis=1).squeeze()
        
        # Mask best to find second best
        S_masked = S_np.copy()
        np.put_along_axis(S_masked, best_idx[:, None], -1, axis=1)
        second_best_vals = np.max(S_masked, axis=1)
        
        # Ratio test: best_dist / second_best_dist < 0.8 
        # or best_sim / second_best_sim > 1.25 (approx)
        # We use similarity: sim1 > sim2 / 0.8
        mask = best_vals > (second_best_vals / ratio_threshold)
        
        # Mutual consistency check (Optional but better)
        # We skip for now to match COLMAP's fast sequential mode.
        
        matches = []
        for i in range(len(mask)):
            if mask[i]:
                matches.append([i, best_idx[i]])
                
        return np.array(matches, dtype=np.int32)

    def write_matches(self, image_id1, image_id2, matches):
        if len(matches) == 0: return
        
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        # COLMAP expects blob of uint32 [idx1, idx2, idx1, idx2, ...]
        blob = matches.astype(np.uint32).tobytes()
        
        cursor = self.db.cursor()
        cursor.execute("INSERT OR REPLACE INTO matches (pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                       (pair_id, matches.shape[0], 2, blob))
        self.db.commit()

    def run_sequential(self, overlap=10):
        print(f"[OxMatcher] Starting Memory-Efficient Sequential Matching (overlap={overlap})...")
        
        cursor = self.db.cursor()
        cursor.execute("SELECT image_id FROM descriptors ORDER BY image_id")
        image_ids = [row[0] for row in cursor.fetchall()]
        
        total_pairs = 0
        start_time = time.time()
        
        # Simple LRU cache for descriptors to avoid redundant SQLite reads
        cache = {}
        def get_cached_desc(img_id):
            if img_id not in cache:
                # Keep cache small (window size)
                if len(cache) > overlap + 2:
                    oldest = min(cache.keys())
                    del cache[oldest]
                cache[img_id] = self.get_descriptor_by_id(img_id)
            return cache[img_id]

        for i in range(len(image_ids)):
            id1 = image_ids[i]
            desc1 = get_cached_desc(id1)
            if desc1 is None: continue
            
            print(f"[OxMatcher] Analyzing Frame {i+1}/{len(image_ids)} (ID:{id1}) against next {overlap} neighbors...")
            
            for j in range(i + 1, min(i + 1 + overlap, len(image_ids))):
                id2 = image_ids[j]
                desc2 = get_cached_desc(id2)
                if desc2 is None: continue
                
                matches = self.match_pair(desc1, desc2)
                self.write_matches(id1, id2, matches)
                total_pairs += 1
                
                # Report status every few pairs
                if total_pairs % 2 == 0:
                    elapsed = time.time() - start_time
                    pairs_per_sec = total_pairs / elapsed
                    remaining = (len(image_ids) * overlap - total_pairs) / pairs_per_sec if pairs_per_sec > 0 else 0
                    print(f"  [Match] GPU ID:{id1} <-> ID:{id2} -> {len(matches)} pairs ({pairs_per_sec:.1f} p/s, approx {remaining/60:.1f}m left)")

        end_time = time.time()
        print(f"[OxMatcher] Finished! Matched {total_pairs} pairs in {end_time - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, required=True)
    parser.add_argument("--overlap", type=int, default=10)
    args = parser.parse_args()
    
    if not os.path.exists(args.database_path):
        print(f"Error: Database {args.database_path} not found.")
    else:
        matcher = OxMatcher(args.database_path)
        matcher.run_sequential(overlap=args.overlap)
