import math

def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    # Clean up predictions format
    predictions = [_.strip().replace(" ","") for _ in predictions]
    
    # Filter predictions not in all_items if provided
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    # Evaluate each batch
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        
        # Pair sequences with scores
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # Sort by scores in descending order
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        
        # Get target items for this batch
        target_items = targets[b]  # Now a list of target items
        
        # Check if each prediction matches any of the target items
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] in target_items:  # Check if prediction is in target items
                one_results.append(1)
            else:
                one_results.append(0)
        
        results.append(one_results)
    
    return results

def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k, normalize=False)  # Return unnormalized hit count
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k, normalize=False)  # Return unnormalized NDCG
        elif m.lower().startswith("mrr"):
            k = int(m.split("@")[1])
            res[m] = mrr_k(topk_results, k, normalize=False)  # Return unnormalized MRR
        elif m.lower().startswith("map"):
            k = int(m.split("@")[1])
            res[m] = map_k(topk_results, k, normalize=False)  # Return unnormalized MAP
        else:
            raise NotImplementedError
    
    return res

def hit_k(topk_results, k, normalize=True):
    hit = 0.0
    total = len(topk_results)
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:  # If any prediction in top-k is correct
            hit += 1
    return hit / total if normalize and total > 0 else hit

def ndcg_k(topk_results, k, normalize=True):
    ndcg = 0.0
    total = len(topk_results)
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            # Calculate DCG
            dcg = 0.0
            for i, r in enumerate(res):
                if r == 1:
                    dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
            
            # Calculate IDCG (ideal case: all relevant items at the beginning)
            idcg = 0.0
            num_relevant = sum(res)
            # Only consider the first k positions for IDCG
            for i in range(min(num_relevant, k)):
                idcg += 1.0 / math.log2(i + 2)
            
            # Add DCG
            ndcg += dcg / idcg if idcg > 0 else 0.0
    
    return ndcg / total if normalize and total > 0 else ndcg

def mrr_k(topk_results, k, normalize=True):
    mrr = 0.0
    total = len(topk_results)
    for row in topk_results:
        res = row[:k]
        # Find the first relevant item
        for i, r in enumerate(res):
            if r == 1:
                mrr += 1.0 / (i + 1)  # Reciprocal rank
                break
    
    return mrr / total if normalize and total > 0 else mrr

def map_k(topk_results, k, normalize=True):
    map_score = 0.0
    total = len(topk_results)
    for row in topk_results:
        res = row[:k]
        # Calculate precision at each position
        precisions = []
        num_relevant = 0
        for i, r in enumerate(res):
            if r == 1:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))
        
        # Calculate average precision
        if precisions:
            map_score += sum(precisions) / len(precisions)
    
    return map_score / total if normalize and total > 0 else map_score

