#!/usr/bin/env python3
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Coreference resolution via a seq2seq transition-based system.

Paper: "Coreference Resolution through a seq2seq Transition-Based System"
       Bohnet, Alberti, Collins — TACL 2023
       https://arxiv.org/abs/2211.12142

Adapted from https://github.com/google-research/google-research/blob/master/coref_mt5/colabs/coreference_decoder.ipynb

Usage
-----
# Run on custom text with a locally-converted HuggingFace checkpoint:
    python coreference.py --model-path coref-mt5-xxl-bohnetetal2022/ --text "Your text here."

# Pass multi-sentence text (one sentence per line) from a file:
    python coreference.py --model-path coref-mt5-xxl-bohnetetal2022/ --file my_doc.txt
"""

import argparse
import logging
import multiprocessing.pool
import sys
import time
from collections import defaultdict
from typing import Optional

import nltk
import torch
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration, T5Tokenizer

from constants import DEFAULT_COREF_MODEL_DIR, MT5_TOKENIZER
from utils import ProgressLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and default values
# ---------------------------------------------------------------------------
# Lazy-loaded tokenizers (initialized on first use to avoid HF Hub requests at import time)
_TOKENIZER_MT5: Optional[T5Tokenizer] = None
TOKENIZER_NLTK = nltk.WordPunctTokenizer()


def _get_tokenizer_mt5() -> T5Tokenizer:
    global _TOKENIZER_MT5
    if _TOKENIZER_MT5 is None:
        _TOKENIZER_MT5 = T5Tokenizer.from_pretrained(MT5_TOKENIZER)
    return _TOKENIZER_MT5


# Token limit defaults
DEFAULT_MAX_TOKEN_LENGTH = 1024
DEFAULT_MAX_NEW_TOKENS = 128

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_coreference_model(hf_model_path: str, compile: bool = True) -> MT5ForConditionalGeneration:
    """Load coreference resolution model weights from HuggingFace checkpoint format.

    Args:
        hf_model_path: Path to a (converted) HuggingFace checkpoint directory.
        compile (bool): If True, compiles the model with torch.compile() before returning.
    """

    logger.info(f"Loading coreference model from {hf_model_path} ...")
    # Use float16 to fit in 64 GB. On Apple Silicon, MPS gives access to
    # the full unified memory pool. On CUDA, device_map="auto" shards
    # across available GPUs.
    if torch.backends.mps.is_available():
        device_map = "mps"
    elif torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = "cpu"
    model = MT5ForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.eval()
    logger.debug(f"Model loaded: device={device_map}, dtype={next(model.parameters()).dtype}")
    if compile:
        logger.debug("Compiling model...")
        return torch.compile(model)
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predictor_fn(batch: list[str],
                 model: MT5ForConditionalGeneration,
                 max_new_tokens: int = None,
                 ) -> list[str]:
    """Run the mT5 model on a single batch of annotated input strings.

    Args:
        batch: List of annotated input strings, e.g.
            ['coref: w | _ The Eiffel Tower ... **']
        max_new_tokens: Maximum number of decoder tokens to generate per
            sequence. Each token requires one decoder forward pass, so smaller
            values are significantly faster. If None (default), computed
            dynamically as max(DEFAULT_MAX_NEW_TOKENS, input_length), so the
            budget always covers at least the input. A warning is logged if
            the limit is reached.

    Returns:
        List of model output strings, one per input, e.g.
            ['It ## is named after -> The Eiffel Tower ## ...']
    """

    # Move inputs to whichever device the model is on (MPS, CUDA, or CPU).
    device = next(model.parameters()).device
    inputs = _get_tokenizer_mt5()(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=DEFAULT_MAX_TOKEN_LENGTH,
    ).to(device)

    # Compute token budget from actual input length when not set explicitly.
    if max_new_tokens is None:
        input_token_length = inputs['input_ids'].shape[1]
        max_new_tokens = round(1.5 * input_token_length)
        logger.debug(f"Input token length: {input_token_length} | Max new tokens: {max_new_tokens}")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,  # NB: greedy search
        )

    # Warn in case of truncated outputs due to reaching maximum new token limit
    generated_len = output_ids.shape[1]
    logger.debug(f"Max generated length this step: {generated_len} / {max_new_tokens} tokens")
    if generated_len >= max_new_tokens:
        logger.warning(
            f"An output sequence reached max_new_tokens={max_new_tokens}. "
            "Coreference links may be truncated."
        )

    return _get_tokenizer_mt5().batch_decode(output_ids, skip_special_tokens=True)


def predict_coreferences(model: MT5ForConditionalGeneration,
                         batches: list[list[str]],
                         max_workers: int = 1,
                         max_new_tokens: int = None,
                         ) -> list[str]:
    """Run prediction over a list of batches, returning one result string per
    input document.

    Args:
        model: Pre-trained and loaded mT5 coreference resolution model.
        batches: List of batches, where each batch is a list of input strings.
        max_workers: Number of worker threads for parallel batch processing.
        max_new_tokens: Decoder token budget per sequence (passed to predictor_fn).
            None means compute dynamically from input length (see predictor_fn).

    Returns:
        Flat list of prediction strings, one per input string.
    """
    if len(batches) == 1:
        return predictor_fn(batches[0], model, max_new_tokens=max_new_tokens)
    with multiprocessing.pool.ThreadPool(max_workers) as pool:
        batch_results = pool.starmap(
            predictor_fn,
            [(batch, model, max_new_tokens) for batch in batches]
        )
    # Flatten: each element of batch_results is a list[str] for one batch.
    return [s for batch in batch_results for s in batch]


# ---------------------------------------------------------------------------
# Coreference decoding helpers
# ---------------------------------------------------------------------------

def normalize_speaker(speaker_in: str) -> str:
    """Wrap speaker name in underscores if not already wrapped."""
    if speaker_in in ("-", "__"):
        return "_"
    speaker = speaker_in.replace(" ", "_").strip()
    if not speaker.startswith("_"):
        speaker = "_" + speaker
    if not speaker.endswith("_"):
        speaker = speaker + "_"
    return speaker


def match_mention_state(
    m: list[str],
    inputs: list[str],
    maps: list[int],
    position=None,
    debug: bool = False,
    start_index: int = 0,
) -> list[tuple[int, int]]:
    """Find token-index spans in `inputs` that match the mention token sequence `m`.

    The mention list uses '##' as a delimiter: tokens before '##' are the
    mention surface form; tokens after are right context used only for
    disambiguation.

    Args:
        m: Mention token list, e.g. ['It', '##', 'is', 'named', 'after'].
        inputs: Full annotated token list to search in.
        maps: Parallel list of original token IDs for each element of `inputs`.
        position: Unused (kept for API compatibility).
        debug: Print extra info when multiple matches found.
        start_index: Start searching from this position in `inputs`.

    Returns:
        List of (start_token_id, end_token_id) spans.
    """
    if "##" in m:
        index_num = m.index("##")
    else:
        if not m[0].startswith("["):
            logger.debug(f"get_chains::error ## not in split: {m}")
        index_num = len(m)

    if "]]" in inputs:
        end_index = inputs.index("]]")
    elif "**" in inputs:
        end_index = inputs.index("**")
    else:
        end_index = len(inputs)

    # Tokens up to (but not including) '##' form the surface form to match.
    m_clean = []
    for x in m:
        if x != "##":
            m_clean.append(x)
        if x == "**":
            break

    # Search for exact match of the surface form in inputs.
    indices = []
    for i in range(start_index, end_index):
        if inputs[i] == m_clean[0] and inputs[i : i + len(m_clean)] == m_clean:
            span = (maps[i], maps[i + index_num - 1])
            indices.append(span)
            if maps[i + index_num - 1] == -1:
                logger.debug(f"index negative {maps[i:]}, index_num {index_num}")
                logger.debug(f"index negative {inputs[i:]}, index_num {index_num}")
                logger.debug(f"i {i} maps_index {i}")

    if len(indices) == 0:
        logger.debug(f"none found match_mention: {m}")
        logger.debug(f"inputs: {inputs}")
        return []
    elif len(indices) > 1 and debug:
        logger.debug(f"match_mention: too many {m} {indices} m_clean - use both")

    if (-1, -1) in indices:
        logger.debug(f"error for {m} {indices}")
        return []

    return indices


def match_link_state(
    link: list[str],
    inputs: list[str],
    maps: list[int],
    cluster_name_to_cluster: dict,
    debug: bool = True,
    node_wise: bool = True,
) -> list:
    """Resolve a coreference link (pair of mention strings) to token-ID spans.

    Args:
        link: List of one or two mention strings (each is a space-separated
            token sequence), e.g. ['It ## is', 'The Eiffel Tower ## is'].
        inputs: Full annotated token list.
        maps: Parallel token-ID list.
        cluster_name_to_cluster: Existing cluster name → span-list mapping.
        debug: Print match details.
        node_wise: If True, a single-mention link is treated as a new mention
            node rather than an error.

    Returns:
        List of resolved span-pair lists.
    """
    link_mentions = [m.split(" ") for m in link]
    links = []

    # Node-wise: single mention — just locate it (no antecedent pairing).
    if len(link_mentions) == 1 and node_wise:
        m0 = link_mentions[0]
        try:
            index_m0 = match_mention_state(m0, inputs, maps, position=None)
            links = [index_m0]
        except Exception as e:
            logger.error(str(e))
        return links

    m0 = link_mentions[0]
    m1 = link_mentions[1]

    logger.debug(f"match_link {m0} {m1}")

    # Antecedent is a cluster reference (e.g. '[1') — use the last mention of
    # that cluster as the antecedent span.
    if m1[0].startswith("[") and len(m1[0]) > 0:
        cluster = cluster_name_to_cluster.get(m1[0], None)
        if cluster is not None:
            index_m1 = [cluster[-1]]
        else:
            logger.debug("cluster does not exist")
            return []
    else:
        index_m1 = match_mention_state(m1, inputs, maps, position=None)

    logger.debug(f"{index_m1} match {m1}")
    if len(index_m1) > 1:
        logger.debug(f"index_m1 {index_m1}")

    try:
        index_m0 = match_mention_state(m0, inputs, maps, position=None)
    except Exception as e:
        logger.error(str(e))
        index_m0 = []

    logger.debug(f'{index_m0} match {m0}')
    if len(index_m0) > 1:
        logger.debug(f"index_m0: {index_m0}")

    if len(index_m1) > 0 and len(index_m0) > 0:
        links.append([index_m1[-1], index_m0[-1]])

    # Keep only the last link when multiple are found.
    if len(links) > 1:
        logger.debug(f"too many links, {links} for link {link}")
        logger.debug(f"context: {inputs}")
        return links[-1:]

    return links


def get_mentions_for_link_state(link: str, node_wise: bool) -> list[str]:
    """Parse a raw link string into a list of one or two mention strings.

    Link format: 'mention0 ## context -> mention1 ## context'
    The '->' separates the anaphor from its antecedent.

    Args:
        link: Raw link string from model output.
        node_wise: If True, a link with no '->' is treated as a single-mention
            node (new mention detection).

    Returns:
        List of one or two stripped mention strings.
    """
    link_split = link.split("->")

    if node_wise and len(link_split) == 1:
        return [link_split[0].strip()]

    if len(link_split) < 2:
        logger.debug(f"link has only one mention - skipping mention: {link}")
        return []

    if len(link_split) > 2:
        logger.debug(f"link has too many mentions - using first two: {link}")

    return [link_split[0].strip(), link_split[1].strip()]


# ---------------------------------------------------------------------------
# State: sentence-by-sentence document state machine
# ---------------------------------------------------------------------------

class State:
    """Sliding-window document state for transition-based coreference.

    The model processes a document one sentence at a time. At each step the
    State object builds an annotated input string:

        coref: <genre> <context with cluster brackets> | <focus sentence> **

    The model predicts coreference links for the focus sentence; the State
    updates its cluster data structures and advances to the next sentence.
    """

    def __init__(
        self,
        input_document: dict,
        node_wise: bool = True,
        max_len_doc: int = DEFAULT_MAX_TOKEN_LENGTH,
    ):
        """
        Args:
            input_document: Dict produced by create_document().
            node_wise: If True, single-mention outputs are treated as new
                mention nodes (enables mention detection alongside linking).
            max_len_doc: Maximum SentencePiece token budget for the input
                string. Context is trimmed from the left when exceeded.
        """
        self.sentence_num = -1
        self.clusters_num = 0

        self.token_map_context: list[int] = []
        self.annotation_context: list[str] = []
        self.annotation_coreference_start: list[list] = []
        self.annotation_coreference_end: list[list] = []
        self.token_map: list[int] = []
        self.annotation: list[str] = []

        # (start_token_id, end_token_id) → cluster span list
        self.mention_index_to_cluster: dict[str, list] = {}
        # (start_token_id, end_token_id) → cluster name string
        self.mention_index_to_cluster_name: dict[str, str] = {}
        # '[N' → cluster span list
        self.cluster_name_to_cluster: dict[str, list] = {}

        self.input_document = input_document
        self.genre = input_document["genres"][0][0]
        self.speakers = {t: spk for (t, spk) in input_document["speakers"]}

        self.done = False
        self.predictions_str: dict[int, str] = {}
        self.node_wise = node_wise
        self.max_len_doc = max_len_doc

        # Advance to the first sentence.
        self.extend()

    def extend_done(self) -> bool:
        return self.done

    def extend(
        self,
        prediction_str: Optional[str] = None,
        use_gold_cluster: bool = False,
        move: bool = True,
    ) -> bool:
        """Process one model prediction and advance to the next sentence.

        Args:
            prediction_str: Model output string for the current focus sentence.
                None on the first call (initialisation).
            use_gold_cluster: Use gold cluster labels (evaluation only).
            move: Always advance to the next sentence after processing.

        Returns:
            True if the document is now fully processed.
        """
        # Commit the current focus tokens into the context window.
        self.token_map_context += self.token_map
        self.annotation_context += self.annotation
        for _ in range(len(self.annotation)):
            self.annotation_coreference_start.append([])
            self.annotation_coreference_end.append([])

        assert len(self.annotation_context) == len(self.annotation_coreference_start)

        self.annotation, self.token_map = [], []

        link_found = False
        if prediction_str is not None and "None [" not in prediction_str:
            links = [l for l in prediction_str.split(";;") if l != ""]

            annotation_update = []
            for link in links:
                link_found = True
                link_mentions = get_mentions_for_link_state(link, self.node_wise)

                if len(link_mentions) < 2 and not (
                    self.node_wise and len(link_mentions)
                ):
                    logger.debug(f"fewer mentions than needed, skip {link_mentions}")
                    continue

                indices = match_link_state(
                    link_mentions,
                    self.annotation_full,
                    self.annotation_full_map,
                    self.cluster_name_to_cluster,
                    debug=False,
                )

                if not indices:
                    logger.debug(f"indices not found {link} {indices}")
                    logger.debug(f"self.annotation_full: {self.annotation_full}")
                    logger.debug(f"annotation + context: {self.get_input_annotation()}",)
                    continue

                index = indices[0]
                cluster: list = []
                for mention_index in index:
                    if str(mention_index) in self.mention_index_to_cluster:
                        cluster = self.mention_index_to_cluster[str(mention_index)]
                        break

                if not cluster:
                    self.clusters_num += 1
                    cluster_name = str(self.clusters_num)

                    if use_gold_cluster:
                        for ni, cx in enumerate(self.input_document["clusters"]):
                            for mx in cx:
                                if mx in index:
                                    cluster_name = str(ni + 1)
                                    break
                else:
                    cluster_name = self.mention_index_to_cluster_name[
                        str(cluster[0])
                    ]

                for mention_index in index:
                    if mention_index not in cluster:
                        cluster.append(mention_index)
                        self.mention_index_to_cluster[str(mention_index)] = cluster
                        self.cluster_name_to_cluster["[" + cluster_name] = cluster
                        self.mention_index_to_cluster_name[
                            str(mention_index)
                        ] = cluster_name
                        annotation_update.append([mention_index, cluster_name])

            # Write cluster start/end markers into the context annotation.
            for update in annotation_update:
                update_index = update[0]
                update_name = update[1]
                for (t, coref_starts, coref_end, tid) in zip(
                    self.annotation_context,
                    self.annotation_coreference_start,
                    self.annotation_coreference_end,
                    self.token_map_context,
                ):
                    if update_index[0] == tid:
                        coref_starts.append(update)
                        coref_starts.sort(key=lambda x: x[0][1], reverse=True)
                    if update_index[1] == tid:
                        coref_end.append("]")

        if move or (prediction_str is not None and "None [" in prediction_str) or not link_found:
            self.sentence_num += 1
            if self.sentence_num not in self.input_document["sentences"]:
                self.done = True
                return True

            tokens = self.input_document["sentences"][self.sentence_num]
            token_map = self.input_document["token_maps"][self.sentence_num]
            first = True
            for tid, t in zip(token_map, tokens):
                if first:
                    self.token_map.append(-1)
                    self.annotation.append(
                        normalize_speaker(self.speakers[tid])
                    )
                    first = False
                self.token_map.append(tid)
                self.annotation.append(t)

        if self.sentence_num not in self.predictions_str:
            self.predictions_str[self.sentence_num] = ""
        if prediction_str is not None:
            self.predictions_str[self.sentence_num] += prediction_str

        return False

    def _build_annotation_full(self) -> None:
        """Rebuild annotation_full and annotation_full_map from current state.

        annotation_full is the full token list for the model input:
            [coref:, genre, <context tokens with cluster brackets>, |,
             <focus tokens>, **]
        annotation_full_map is the parallel list of original token IDs
        (-1 for special tokens / brackets).
        """
        self.annotation_full = ["coref:", self.genre]
        self.annotation_full_map = [-1, -1]

        for t, coref_starts, coref_end, tid in zip(
            self.annotation_context,
            self.annotation_coreference_start,
            self.annotation_coreference_end,
            self.token_map_context,
        ):
            for coref_start in coref_starts:
                self.annotation_full.append("[" + coref_start[-1])
                self.annotation_full_map.append(-1)
            self.annotation_full.append(t)
            self.annotation_full_map.append(tid)
            for _ in coref_end:
                self.annotation_full.append("]")
                self.annotation_full_map.append(-1)

        self.annotation_full += ["|"] + self.annotation
        self.annotation_full_map += [-1] + self.token_map
        self.annotation_full += ["**"]
        self.annotation_full_map += [-1]

    # Keep original name as an alias so State.extend() can call it.
    input_annotation = _build_annotation_full

    def _encode(self, annotation_str: str) -> list[int]:
        return _get_tokenizer_mt5().encode(annotation_str)

    def get_input_annotation(self, context_right: bool = True) -> str:
        """Return the model input string for the current focus sentence.

        Trims left context if the encoded length exceeds max_len_doc, and
        optionally appends right-context sentences that fit within the budget.

        Args:
            context_right: Append future sentences as right context if space
                allows.

        Returns:
            Annotated input string ready to feed to the model.
        """
        self._build_annotation_full()
        annotation_str = " ".join(self.annotation_full)

        enc = self._encode(annotation_str)
        shorten = len(enc) > self.max_len_doc

        # Trim left context until the input fits within the token budget.
        while len(enc) > self.max_len_doc:
            self.annotation_context = self.annotation_context[1:]
            self.token_map_context = self.token_map_context[1:]
            self.annotation_coreference_start = self.annotation_coreference_start[1:]
            self.annotation_coreference_end = self.annotation_coreference_end[1:]
            self._build_annotation_full()
            annotation_str = " ".join(self.annotation_full)
            enc = self._encode(annotation_str)

        self.annotation_context_right: list[str] = []

        # Append future sentences as right context if budget allows.
        if not shorten and context_right:
            total_len = len(enc)
            sentence_num = self.sentence_num

            while True:
                sentence_num += 1
                if sentence_num not in self.input_document["sentences"]:
                    break

                first = True
                next_tokens: list[str] = []
                for t, tid in zip(
                    self.input_document["sentences"][sentence_num],
                    self.input_document["token_maps"][sentence_num],
                ):
                    if first:
                        next_tokens.append(
                            normalize_speaker(self.speakers[tid])
                        )
                        first = False
                    next_tokens.append(t)

                candidate_right = self.annotation_context_right + next_tokens
                if len(self._encode(" ".join(candidate_right))) + total_len > self.max_len_doc:
                    break
                self.annotation_context_right = candidate_right

            if self.annotation_context_right:
                annotation_str = (
                    annotation_str + " " + " ".join(self.annotation_context_right)
                )

        enc = self._encode(annotation_str)
        if len(enc) > self.max_len_doc:
            logger.warning(f"Document too long: {len(enc)}")

        return annotation_str


# ---------------------------------------------------------------------------
# Document utilities
# ---------------------------------------------------------------------------

def create_document(document: str,
                    title: str = "document",
                    is_tokenized: bool = False,
                    ) -> dict:
    """Tokenise a plain-text document into the internal input format.

    Args:
        document: Text with sentences separated by newlines ('\\n').
        title: Document identifier used in output.
        is_tokenized (bool):    Indicates whether the input text is already tokenized.
                                If False, the text is tokenized here using NLTK.

    Returns:
        Dict with keys: doc_key, sentences, token_maps, speakers, genres.
        'sentences' and 'token_maps' are dicts keyed by sentence index.
    """
    input_document: dict = {
        "doc_key": title,
        "sentences": {},
        "token_maps": {},
        "speakers": [],
        "genres": [],
    }

    tid = 0
    for k, sentence in enumerate(document.split("\n")):
        if is_tokenized:
            input_document["sentences"][k] = sentence.split()
        else:
            input_document["sentences"][k] = TOKENIZER_NLTK.tokenize(text=sentence)
        input_document["token_maps"][k] = []
        for _ in input_document["sentences"][k]:
            input_document["token_maps"][k].append(tid)
            input_document["speakers"].append((tid, "_"))
            input_document["genres"].append("wi")
            tid += 1

    return input_document


def create_next_batch(
    states_dict: dict,
    batch_size: int = 1,
    num_batches: int = 1,
) -> tuple[list[State], list[list[str]]]:
    """Build the next batch of model inputs from active document states.

    Args:
        states_dict: Mapping of document key → State.
        batch_size: Number of documents per batch.
        num_batches: Number of batches to create.

    Returns:
        (states, batches) where states[i] is the State object whose input
        string is batches[i//batch_size][i%batch_size].
    """
    batches: list[list[str]] = [[]]
    states: list[State] = []

    # Sort active states by accumulated context length so that documents at
    # similar stages of processing land in the same batch.
    # This minimises padding in the GPU forward pass (mixed short/long inputs 
    # cause the whole batch to be padded to the longest, which dominates compute time).
    active = sorted(
        (s for s in states_dict.values() if not s.extend_done()),
        key=lambda s: len(s.annotation_context),
    )
    for state in active[:batch_size * num_batches]:
        states.append(state)
        batches[-1].append(state.get_input_annotation())
        if len(batches[-1]) >= batch_size:
            if len(batches) >= num_batches:
                break
            batches.append([])

    return states, batches


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------
def render_annotated_text(states_dict: dict) -> dict[str, dict]:
    """Format coreference-annotated output text (cluster brackets with entity index are inserted around each mention)
    for each document and return mention token spans per entity.



    Args:
        states_dict: Mapping of document key → State

    Example output:
    {
        "doc1": {
            "text": "[1 The Eiffel Tower ] ... [1 it ] ...",
            "spans": {
                1: [(1, 3), (8, 8)],
            },
        }
    }
    """

    outputs = {}

    for doc_name, s in states_dict.items():
        pred_clusters = list(s.cluster_name_to_cluster.values())

        # Reconstruct flat token list and token-ID list.
        text = []
        text_map = []

        for k in sorted(s.input_document["sentences"]):
            text += s.input_document["sentences"][k]
            text_map += s.input_document["token_maps"][k]

        # Map original token IDs -> flat token indices
        tid_to_pos = {tid: pos for pos, tid in enumerate(text_map)}

        # For each token position, collect which cluster spans start/end here.
        cluster_starts = [[] for _ in text_map]
        cluster_ends = [[] for _ in text_map]
        entity_spans = defaultdict(list)

        for ci in pred_clusters:
            for m in ci:
                start_tid, end_tid = m

                if start_tid not in tid_to_pos or end_tid not in tid_to_pos:
                    continue

                start_pos = tid_to_pos[start_tid]
                end_pos = tid_to_pos[end_tid]

                m_len = end_tid - start_tid
                name = s.mention_index_to_cluster_name[str(m)]

                # Store span
                entity_spans[name].append((start_pos, end_pos))

                # Annotation bookkeeping
                cluster_starts[start_pos].append((name, m_len))
                cluster_ends[end_pos].append("]")

        # Build the annotated token sequence.
        all_text: list[str] = []
        for tok, starts, ends in zip(text, cluster_starts, cluster_ends):
            # Longer spans open first so brackets nest correctly.
            for name, _ in sorted(starts, key=lambda x: x[1], reverse=True):
                all_text.append("[" + str(name))
            all_text.append(tok)
            if ends:
                all_text.append("".join(ends))

        outputs[doc_name] = {
            "text": " ".join(all_text),
            "spans": dict(entity_spans),
        }

    return outputs


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(
    documents: dict[str, str],
    model: MT5ForConditionalGeneration,
    batch_size: int = 8,
    max_workers: int = 1,
    max_new_tokens: int = None,
    is_tokenized: bool = False,
    debug: bool = False,
) -> dict[str, str]:
    """Process one or more documents through the coreference system.

    Args:
        documents: Mapping of document title → plain text (sentences separated
            by newlines).
        model: Pre-trained and loaded mT5 model for coreference resolution.
        batch_size: Number of documents packed into a single GPU forward pass.
        max_new_tokens: Decoder token budget per sequence. Each additional token
            costs one decoder forward pass. None (default) means compute
            dynamically from input length. A warning is logged if any output
            hits the limit.
        debug: Print per-step model inputs and predictions.

    Returns:
        Mapping of document title → annotated text string with cluster brackets.
    """
    states_dict = {
        title: State(create_document(text, title, is_tokenized=is_tokenized))
        for title, text in documents.items()
    }

    total_time = time.time()
    total_results = 0
    total_sentences = sum(len(s.input_document["sentences"]) for s in states_dict.values())
    n_docs = len(states_dict)
    max_sents = max(len(s.input_document["sentences"]) for s in states_dict.values())

    # Periodic log fallback when stderr is not a tty (progress bar disabled)
    progress = ProgressLogger(
        total=total_sentences,
        label="Coreference",
        unit="sent",
        enabled=not sys.stderr.isatty()
    )
    sentences_processed = 0

    with tqdm(total=total_sentences, unit="sent", desc="Coreference", disable=not sys.stderr.isatty()) as pbar:
        step = 0
        while True:
            t = time.time()
            states, batches = create_next_batch(
                states_dict,
                batch_size=batch_size,
            )

            if not states:
                break

            if debug:
                doc_keys = {s.input_document["doc_key"] for s in states}
                logger.debug(f"Step {step} — processing: {doc_keys}")

            # Get prediction strings
            results = predict_coreferences(
                model,
                batches,
                max_new_tokens=max_new_tokens,
                max_workers=max_workers,
            )

            flat_inputs = [s for batch in batches for s in batch]
            for state, result, input_str in zip(states, results, flat_inputs):
                state.extend(result)
                logger.debug(f"\tinput:\t{input_str}")
                logger.debug(f"\tprediction:\t{result}")

            total_results += len(results)
            elapsed = time.time() - t
            logger.debug(
                f"  step time: {elapsed:.3f}s | "
                f"per-seq: {elapsed/len(results):.3f}s | "
                f"total per-seq: {(time.time()-total_time)/total_results:.3f}s"
            )

            pbar.update(len(states))
            sentences_processed += len(states)
            docs_done = sum(1 for s in states_dict.values() if s.extend_done())
            progress.update(len(states), extra=f"{docs_done}/{n_docs} docs done")

            # pbar.n // n_docs approximates which sentence pass reached across all docs
            sent_round = pbar.n // n_docs
            pbar.set_postfix({"sent": f"{sent_round}/{max_sents}", "docs done": f"{docs_done}/{n_docs}"})
            step += 1

    return render_annotated_text(states_dict)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coreference resolution with mT5 (seq2seq transition-based)."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_COREF_MODEL_DIR,
        help="Path to a HuggingFace checkpoint directory for a pre-trained mT5 coreference resolution model.",
    )
    parser.add_argument(
        "--text",
        help="Input text(s) (texts separated by newlines)."
    )
    parser.add_argument(
        "--file",
        nargs="*",
        help="Path to a plain-text file (one sentence per line)."
    )
    parser.add_argument(
        "--file-encoding",
        default="utf-8",
        help="Encoding for input text files."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of documents packed into a single GPU forward pass for coreference resolution"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum decoder tokens to generate per sequence per step. "
             "Each token costs one decoder forward pass. "
             "Default: computed dynamically from input length. "
             "A warning is logged if any output reaches this limit."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of worker threads for parallel processing."
    )
    parser.add_argument(
        "--pretokenized",
        action="store_true",
        help="Marks the input text(s) as pretokenized and will not tokenize again.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-step model inputs and predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.file and args.text:
        logger.error("Provide only --text or --file argument, not both.")
        return
    if args.file:
        documents = {}
        for filepath in args.file:
            with open(filepath, 'r', encoding=args.file_encoding) as f:
                text = f.read()
                documents[filepath] = text
    elif args.text:
        texts = args.text.replace("\\n", "\n").split("\n")
        documents = {
            f"doc_{i + 1:03d}": text
            for i, text in enumerate(texts)
        }
    else:
        logger.error("Provide --text or --file argument.")
        return
    logger.debug(f"Loaded {len(documents)} input document(s)")

    coref_model = load_coreference_model(hf_model_path=args.model_path)
    annotated = run(
        documents=documents,
        model=coref_model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_new_tokens=args.max_new_tokens,
        is_tokenized=args.pretokenized,
        debug=args.debug,
    )

    for doc_name, coref_result in annotated.items():
        print(f"\n=== {doc_name} ===")
        print(coref_result["text"])


if __name__ == "__main__":
    main()
