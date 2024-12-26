import numpy as np
from ranx import Qrels, Run, evaluate

def ranx(k, metrics, query_labels, candidate_labels, results):
    qrels_dict = {}
    run_dict = {}

    for i, query_label in enumerate(query_labels):
        query_id = f"q{i}"
        relevance = {}
        scores = {}

        for rank, idx in enumerate(results[i][:k]):
            doc_id = f"d{idx}"
            relevance[doc_id] = 1 if np.array_equal(candidate_labels[idx], query_label) else 0
            # scores[doc_id] = 1.0 / (D[i][rank] + 1e-6) # L2距離のみ正常に動作、cos類似度だとランキングが逆になる
            scores[doc_id] = 100 - rank
        
        qrels_dict[query_id] = relevance
        run_dict[query_id] = scores
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    metrics = evaluate(qrels, run, metrics)

    for metric, value in metrics.items():
        print(f"{metric}: {value:4f}")

def evaluate_search(results, query_loader, candidate_loader): # results: I(検索結果のインデックス)[50,100]
    query_labels = np.vstack([label for _, _, label in query_loader]) # [50, 14]
    candidate_labels = np.vstack([label for _, _, label in candidate_loader]) # [1000, 14]

    """
    ranxを使用
    """
    ranx(10, ["map@10", "precision@1", "precision@5"], query_labels, candidate_labels, results)
    ranx(100, ["precision@10", "precision@100"], query_labels, candidate_labels, results)

    return None