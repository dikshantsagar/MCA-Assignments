import numpy  as np
from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    alpha = 0.8
    beta = 0.1
    epochs = 3

    for e in range(epochs):
        
        for i in range(vec_queries.shape[0]):
            docs = sim[:,i]
            indices = np.argsort(-docs)
            rel = np.array([vec_docs[j] for j in indices[:n]])
            nonrel = np.array([vec_docs[j] for j in indices[-n:]])
            term1 = alpha * np.sum(rel)/n
            term2 = beta * np.sum(nonrel)/n

            vec_queries[i] = vec_queries[i] + term1 - term2


    rf_sim = cosine_similarity(vec_docs,vec_queries)
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    alpha = 0.8
    beta = 0.1
    epochs = 3
    N=10
    for e in range(epochs):
        
        for i in range(vec_queries.shape[0]):
            docs = sim[:,i]
            indices = np.argsort(-docs)
            rel = np.array([vec_docs[j] for j in indices[:n]])
            nonrel = np.array([vec_docs[j] for j in indices[-n:]])
            term1 = alpha * np.sum(rel)/n
            term2 = beta * np.sum(nonrel)/n
            reldocwords = []
            tempvar = tfidf_model.inverse_transform(rel[0])

            for k in range(N):
                reldocwords.append(tempvar[0][k])

            queryexp = tfidf_model.transform([' '.join(reldocwords)])
            
            vec_queries[i] = vec_queries[i] + term1 - term2 + queryexp 

    rf_sim = cosine_similarity(vec_docs,vec_queries)
    return rf_sim