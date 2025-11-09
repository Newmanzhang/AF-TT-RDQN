from __future__ import annotations
from typing import List, Tuple
import numpy as np
from constants import set_system_size
set_system_size(5, 3, 2)
from constants import M, N, K, Action

PAD_ID      = 0
SHIP_OFFSET = 1
EDGE_OFFSET = lambda: SHIP_OFFSET + M
CH_OFFSET   = lambda: EDGE_OFFSET() + N
LOCAL_ID    = lambda: CH_OFFSET() + K
VOCAB_SIZE  = lambda: LOCAL_ID() + 1


def action2tokens_and_mask(act: Action, pad_id: int = PAD_ID) -> Tuple[List[int], np.ndarray]:

    tokens: List[int] = []
    ship_token_spans = []
    for i, (node, chs) in enumerate(act,1):

        cur = []
        cur.append(SHIP_OFFSET + (i -1) )

        if node == 0:
            cur.append(LOCAL_ID())
        else:

            cur.append(EDGE_OFFSET() + (node - 1))

            for ch in sorted(chs):
                cur.append(CH_OFFSET() + (ch - 1))
        while len(cur) < 2 + K:
            cur.append(pad_id)
        ship_token_spans.append((len(tokens), len(tokens) + len(cur)))
        tokens.extend(cur)

    total_len = M * (2 + K)
    tokens = tokens[:total_len]

    masks = np.zeros((M, total_len), dtype=bool)
    for i, (start, end) in enumerate(ship_token_spans):
        masks[i, start:end] = True
    return tokens, masks


def batch_action2tokens_and_masks(acts: List[Action],  pad_id: int = PAD_ID):
    token_list = []
    mask_list = []
    for act in acts:
        t, m = action2tokens_and_mask(act, pad_id)
        token_list.append(t)
        mask_list.append(m)
    return np.stack(token_list, axis=0), np.stack(mask_list, axis=0)


def get_vocab_size() -> int:
    return VOCAB_SIZE()

def get_pad_id() -> int:
    return PAD_ID
