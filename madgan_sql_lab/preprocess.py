# Author: William Kandolo
import sqlparse
from sqlparse.tokens import Token
from collections import defaultdict

token_to_idx = defaultdict(lambda: len(token_to_idx))
token_to_idx["<PAD>"] = 0

def normalize_sql(sql):
    tokens = sqlparse.parse(sql)[0].flatten()
    normalized = []
    for token in tokens:
        if token.ttype in Token.Literal.Number:
            normalized.append("<NUM>")
        elif token.ttype in Token.Literal.String.Single:
            normalized.append("<STR>")
        elif not token.is_whitespace:
            normalized.append(token.value.upper())
    return normalized

def vectorize_sequence(tokens, max_len=20):
    idx_seq = [token_to_idx[token] for token in tokens]
    if len(idx_seq) < max_len:
        idx_seq += [0] * (max_len - len(idx_seq))
    return idx_seq[:max_len]
