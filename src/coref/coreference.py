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
# Run the built-in Eiffel Tower demo (no model needed):
    python coreference.py --demo

# Run on custom text with a locally-converted HuggingFace checkpoint:
    python coreference.py --model-path coref-mt5-xxl-bohnetetal2022/ --text "Your text here."

# Pass multi-sentence text (one sentence per line) from a file:
    python coreference.py --model-path coref-mt5-xxl-bohnetetal2022/ --file my_doc.txt
"""

import argparse
import multiprocessing.pool
import time
from typing import Optional

import nltk
from transformers import T5Tokenizer, MT5ForConditionalGeneration
import torch


# ---------------------------------------------------------------------------
# Demo predictions (hardcoded from the notebook — Eiffel Tower Wikipedia excerpt)
# ---------------------------------------------------------------------------

DEMO_TEXT_TITLE = "Eiffel Tower Wiki"
DEMO_TEXT = (
    "The Eiffel Tower (French: tour Eiffel) is a wrought-iron lattice tower "
    "on the Champ de Mars in Paris, France. It is named after the engineer "
    "Gustave Eiffel, whose company designed and built the tower.\n"
    "Locally nicknamed \"La dame de fer\" (French for \"Iron Lady\"), it was "
    "constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.\n"
    "Although initially criticised by some of France's leading artists and "
    "intellectuals for its design, it has since become a global cultural icon "
    "of France and one of the most recognisable structures in the world.\n"
    "The Eiffel Tower is the most visited monument with an entrance fee in "
    "the world: 6.91 million people ascended it in 2015.\n"
    "It was designated a monument historique in 1964, and was named part of a "
    "UNESCO World Heritage Site (\"Paris, Banks of the Seine\") in 1991."
)

# One prediction string per sentence (separated by ';;', each link is
# 'mention ## context -> antecedent ## context').
DEMO_PREDICTIONS = [
    (
        'It ## is named after -> The Eiffel Tower ( French : tour Eiffel ) ##'
        ' is a wrought ;; the tower ## . ** _ -> It ## is named after ;;'
    ),
    'it ## was constructed from -> [1 ;;',
    (
        'its ## design , it -> [1 ;; it ## has since become -> its ## design'
        " , it ;; France ' s ## leading artists and -> France ## . [1 It ;;"
        " France ## and one of -> France ' s ## leading artists and ;;"
    ),
    (
        'The Eiffel Tower ## is the most -> [1 ;; it ## in 2015 . -> The'
        ' Eiffel Tower ## is the most ;; the world ## : 6 . -> the world ## .'
        ' | _ ;;'
    ),
    (
        'It ## was designated a -> [1 ;; Paris ## , Banks of -> Paris , [2'
        ' France ## ] . [1 ;;'
    ),
]


# ---------------------------------------------------------------------------
# Globals set by load_model()
# ---------------------------------------------------------------------------

tokenizer_mt5: Optional[T5Tokenizer] = None
model: Optional[MT5ForConditionalGeneration] = None
tokenizer_nltk = nltk.WordPunctTokenizer()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(hf_model_path: Optional[str] = None) -> None:
    """Load the tokenizer (always from HuggingFace Hub) and optionally the
    converted HuggingFace model weights.

    Args:
        hf_model_path: Path to a converted HuggingFace checkpoint directory.
            Leave None to use demo mode (no model needed).
    """
    global tokenizer_mt5, model

    print("Loading tokenizer from google/mt5-xxl ...")
    tokenizer_mt5 = T5Tokenizer.from_pretrained("google/mt5-xxl")

    if hf_model_path:
        print(f"Loading model from {hf_model_path} ...")
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
        print(f"Model loaded — device={device_map}, "
              f"dtype={next(model.parameters()).dtype}")
    else:
        print("No model path provided — running in demo mode.")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predictor_fn(batch: list[str]) -> list[str]:
    """Run the mT5 model on a single batch of annotated input strings.

    Args:
        batch: List of annotated input strings, e.g.
            ['coref: w | _ The Eiffel Tower ... **']

    Returns:
        List of model output strings, one per input, e.g.
            ['It ## is named after -> The Eiffel Tower ## ...']
    """
    if model is None:
        raise RuntimeError(
            "Model is not loaded. Call load_model(hf_model_path=...) first, "
            "or run with --demo to use hardcoded predictions."
        )

    # Move inputs to whichever device the model is on (MPS, CUDA, or CPU).
    device = next(model.parameters()).device
    inputs = tokenizer_mt5(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=1,
        )

    return tokenizer_mt5.batch_decode(output_ids, skip_special_tokens=True)


def predict_coreferences(batches: list[list[str]], threads_to_use: int = 1) -> list[str]:
    """Run prediction over a list of batches, returning one result string per
    input document.

    Args:
        batches: List of batches, where each batch is a list of input strings.
        threads_to_use: Number of threads for parallel batch processing.

    Returns:
        Flat list of prediction strings, one per input string.
    """
    with multiprocessing.pool.ThreadPool(threads_to_use) as pool:
        batch_results = pool.map(predictor_fn, batches)
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
            print("get_chains::error ## not in split", m)
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
                print("index negative", maps[i:], " index_num", index_num)
                print("index negative", inputs[i:], " index_num", index_num)
                print(f"i {i} maps_index {i}")

    if len(indices) == 0:
        print("none found match_mention", m)
        print("inputs", inputs)
        return []
    elif len(indices) > 1 and debug:
        print("match_mention: too many ", m, indices, "m_clean - use both")

    if (-1, -1) in indices:
        print("error for ", m, indices)
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
            print(str(e))
        return links

    m0 = link_mentions[0]
    m1 = link_mentions[1]

    if debug:
        print("match_link", m0, m1)

    # Antecedent is a cluster reference (e.g. '[1') — use the last mention of
    # that cluster as the antecedent span.
    if m1[0].startswith("[") and len(m1[0]) > 0:
        cluster = cluster_name_to_cluster.get(m1[0], None)
        if cluster is not None:
            index_m1 = [cluster[-1]]
        else:
            print("cluster does not exist")
            return []
    else:
        index_m1 = match_mention_state(m1, inputs, maps, position=None)

    if debug:
        print(index_m1, "match", m1)
    if len(index_m1) > 1:
        print("index_m1", index_m1)

    try:
        index_m0 = match_mention_state(m0, inputs, maps, position=None)
    except Exception as e:
        print("error", str(e))
        index_m0 = []

    if debug:
        print(index_m0, "match", m0)
    if len(index_m0) > 1:
        print("index_m0", index_m0)

    if len(index_m1) > 0 and len(index_m0) > 0:
        links.append([index_m1[-1], index_m0[-1]])

    # Keep only the last link when multiple are found.
    if len(links) > 1:
        print("too many links,", links, "for link", link)
        print("context", inputs)
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
        print("link has only one mention - skipping mention", link)
        return []

    if len(link_split) > 2:
        print("link has too many mentions - using first two.", link)

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
        max_len_doc: int = 3000,
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
                    print("less mentions than needed, skip", link_mentions)
                    continue

                indices = match_link_state(
                    link_mentions,
                    self.annotation_full,
                    self.annotation_full_map,
                    self.cluster_name_to_cluster,
                    debug=False,
                )

                if not indices:
                    print("not found !!")
                    print("indices not found", link, indices)
                    print("self.annotation_full", self.annotation_full)
                    print("annotation + context", self.get_input_annotation())
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
        return tokenizer_mt5.encode(annotation_str)

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
            print("warning: document too long", len(enc))

        return annotation_str


# ---------------------------------------------------------------------------
# Document utilities
# ---------------------------------------------------------------------------

def create_document(document: str, title: str = "not_named") -> dict:
    """Tokenise a plain-text document into the internal input format.

    Args:
        document: Text with sentences separated by newlines ('\\n').
        title: Document identifier used in output.

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
        input_document["sentences"][k] = tokenizer_nltk.tokenize(text=sentence)
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

    for key, state in states_dict.items():
        if state.extend_done():
            continue
        states.append(state)
        if len(states) >= batch_size * num_batches:
            break

    for state in states:
        batches[-1].append(state.get_input_annotation())
        if len(batches[-1]) >= batch_size:
            if len(batches) >= num_batches:
                break
            batches.append([])

    return states, batches


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def render_annotated_text(states_dict: dict) -> dict[str, str]:
    """Format coreference-annotated output text for each document.

    Cluster brackets are inserted around each mention:
        [1 The Eiffel Tower ] ... [1 it ] ...

    Args:
        states_dict: Mapping of document key → State (after full processing).

    Returns:
        Mapping of document key → annotated text string.
    """
    outputs: dict[str, str] = {}

    for doc_name, s in states_dict.items():
        pred_clusters = list(s.cluster_name_to_cluster.values())

        # Reconstruct flat token list and token-ID list.
        text: list[str] = []
        text_map: list[int] = []
        for k in sorted(s.input_document["sentences"]):
            text += s.input_document["sentences"][k]
            text_map += s.input_document["token_maps"][k]

        # For each token position, collect which cluster spans start/end here.
        cluster_starts: list[list[tuple[str, int]]] = [[] for _ in text_map]
        cluster_ends: list[list[str]] = [[] for _ in text_map]

        for ci in pred_clusters:
            for m in ci:
                for pos, tid in enumerate(text_map):
                    if tid == m[0]:
                        m_len = m[1] - m[0]
                        name = s.mention_index_to_cluster_name[str(m)]
                        cluster_starts[pos].append((name, m_len))
                    if tid == m[1]:
                        cluster_ends[pos].append("]")

        # Build the annotated token sequence.
        all_text: list[str] = []
        for tok, starts, ends in zip(text, cluster_starts, cluster_ends):
            # Longer spans open first so brackets nest correctly.
            for name, _ in sorted(starts, key=lambda x: x[1], reverse=True):
                all_text.append("[" + str(name))
            all_text.append(tok)
            if ends:
                all_text.append("".join(ends))

        outputs[doc_name] = " ".join(all_text)

    return outputs


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(
    documents: dict[str, str],
    demo_predictions: Optional[list[str]] = None,
    debug: bool = False,
) -> dict[str, str]:
    """Process one or more documents through the coreference system.

    Args:
        documents: Mapping of document title → plain text (sentences separated
            by newlines).
        demo_predictions: If provided, use these hardcoded prediction strings
            instead of calling the model (one string per sentence step).
        debug: Print per-step input/output strings.

    Returns:
        Mapping of document title → annotated text string with cluster brackets.
    """
    states_dict = {
        title: State(create_document(text, title))
        for title, text in documents.items()
    }

    total_time = time.time()
    total_results = 0

    for step in range(1000):
        t = time.time()
        states, batches = create_next_batch(states_dict)

        if not states:
            break

        if debug:
            doc_keys = {s.input_document["doc_key"] for s in states}
            print(f"\nStep {step} — processing: {doc_keys}")

        # Get prediction strings: either from demo list or from the live model.
        if demo_predictions is not None:
            if step >= len(demo_predictions):
                print(f"Warning: ran out of demo predictions at step {step}.")
                break
            results = [demo_predictions[step]]
        else:
            results = predict_coreferences(batches, threads_to_use=len(batches))

        for state, result, batch in zip(states, results, batches):
            state.extend(result)
            if debug:
                print("  input:      ", batch[0] if batch else "")
                print("  prediction: ", result)

        total_results += len(results)
        elapsed = time.time() - t
        if debug:
            print(
                f"  step time: {elapsed:.3f}s | "
                f"per-seq: {elapsed/len(results):.3f}s | "
                f"total per-seq: {(time.time()-total_time)/total_results:.3f}s"
            )

    return render_annotated_text(states_dict)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coreference resolution with mT5 (seq2seq transition-based)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in Eiffel Tower demo (no model required).",
    )
    mode.add_argument(
        "--model-path",
        metavar="PATH",
        help=(
            "Path to a converted HuggingFace checkpoint directory. "
            "See module docstring for conversion instructions."
        ),
    )
    parser.add_argument(
        "--text",
        metavar="TEXT",
        help="Input text (sentences separated by newlines). "
             "Ignored when --demo is set.",
    )
    parser.add_argument(
        "--file",
        metavar="FILE",
        help="Path to a plain-text file (one sentence per line). "
             "Ignored when --demo is set.",
    )
    parser.add_argument(
        "--title",
        default="document",
        help="Document title used in output (default: 'document').",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-step model inputs and predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        # Demo mode: use the Eiffel Tower example with hardcoded predictions.
        # The tokenizer is still needed for context-window length checks inside
        # the State object, so we load it even in demo mode.
        load_model(hf_model_path=None)
        annotated = run(
            documents={DEMO_TEXT_TITLE: DEMO_TEXT},
            demo_predictions=DEMO_PREDICTIONS,
            debug=args.debug,
        )
    else:
        # Live model mode.
        if args.file:
            with open(args.file, encoding="utf-8") as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            print("Error: provide --text or --file when using --model-path.")
            return

        load_model(hf_model_path=args.model_path)
        annotated = run(
            documents={args.title: text},
            demo_predictions=None,
            debug=args.debug,
        )

    for doc_name, text in annotated.items():
        print(f"\n=== {doc_name} ===")
        print(text)


if __name__ == "__main__":
    main()
