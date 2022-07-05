from typing import Optional, Callable, Iterable, Iterator
from pathlib import Path

import random
import itertools
import spacy
import warnings
import functools
from spacy.training import Corpus, Example
from spacy.language import Language
from spacy.kb import KnowledgeBase as SpacyKnowledgeBase, Candidate
from spacy.tokens import Span

from scispacy.custom_tokenizer import combined_rule_tokenizer
from scispacy.data_util import read_full_med_mentions, read_ner_from_tsv
from scispacy.candidate_generation import CandidateGenerator


def iter_sample(iterable: Iterable, sample_percent: float) -> Iterator:
    for item in iterable:
        if len(item.reference) == 0:
            continue
        coin_flip = random.uniform(0, 1)
        if coin_flip < sample_percent:
            yield item


@spacy.registry.callbacks("replace_tokenizer")
def replace_tokenizer_callback() -> Callable[[Language], Language]:
    def replace_tokenizer(nlp: Language) -> Language:
        nlp.tokenizer = combined_rule_tokenizer(nlp)
        return nlp

    return replace_tokenizer


@spacy.registry.readers("parser_tagger_data")
def parser_tagger_data(
    path: Path,
    mixin_data_path: Optional[Path],
    mixin_data_percent: float,
    gold_preproc: bool,
    max_length: int = 0,
    limit: int = 0,
    augmenter: Optional[Callable] = None,
    seed: int = 0,
) -> Callable[[Language], Iterator[Example]]:
    random.seed(seed)
    main_corpus = Corpus(
        path,
        gold_preproc=gold_preproc,
        max_length=max_length,
        limit=limit,
        augmenter=augmenter,
    )
    if mixin_data_path is not None:
        mixin_corpus = Corpus(
            mixin_data_path,
            gold_preproc=gold_preproc,
            max_length=max_length,
            limit=limit,
            augmenter=augmenter,
        )

    def mixed_corpus(nlp: Language) -> Iterator[Example]:
        if mixin_data_path is not None:
            main_examples = main_corpus(nlp)
            mixin_examples = iter_sample(mixin_corpus(nlp), mixin_data_percent)
            return itertools.chain(main_examples, mixin_examples)
        else:
            return main_corpus(nlp)

    return mixed_corpus


@spacy.registry.readers("med_mentions_reader")
def med_mentions_reader(
    directory_path: str, split: str
) -> Callable[[Language], Iterator[Example]]:
    train, dev, test = read_full_med_mentions(
        directory_path, label_mapping=None, span_only=True, spacy_format=True
    )

    def corpus(nlp: Language) -> Iterator[Example]:
        if split == "train":
            original_examples = train
        elif split == "dev":
            original_examples = dev
        elif split == "test":
            original_examples = test
        else:
            raise Exception(f"Unexpected split {split}")

        for original_example in original_examples:
            # import ipdb

            # ipdb.set_trace()
            doc = nlp.make_doc(original_example[0])
            if len(doc) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                try:
                    spacy_example = Example.from_dict(doc, original_example[1])
                except:
                    # remove the example if it does not align with the tokenization
                    mistokenizations_to_delete = []
                    for (start_char, end_char) in original_example[1]["links"].keys():
                        if doc.char_span(start_char, end_char) is None:
                            mistokenizations_to_delete.append((start_char, end_char))
                    for (start_char, end_char) in mistokenizations_to_delete:
                        del original_example[1]["links"][(start_char, end_char)]
                        original_example[1]["entities"].remove(
                            (start_char, end_char, "ENTITY")
                        )
                    spacy_example = Example.from_dict(doc, original_example[1])
            yield spacy_example

    return corpus


@spacy.registry.readers("specialized_ner_reader")
def specialized_ner_reader(file_path: str):
    original_examples = read_ner_from_tsv(file_path)

    def corpus(nlp: Language):
        for original_example in original_examples:
            doc = nlp.make_doc(original_example[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                spacy_example = Example.from_dict(doc, original_example[1])
            yield spacy_example

    return corpus


def scispacy_get_candidates(
    scispacy_candidate_generator: CandidateGenerator,
    spacy_kb: SpacyKnowledgeBase,
    span: Span,
) -> Iterator[Candidate]:
    scispacy_candidates = sorted(
        scispacy_candidate_generator([span.text], k=1)[0],
        key=lambda x: max(x.similarities),
        reverse=True,
    )
    for scispacy_candidate in scispacy_candidates:
        # limited to UMLS entries with definitions
        if not spacy_kb.contains_entity(scispacy_candidate.concept_id):
            continue

        entity_hash = spacy_kb.vocab.strings[scispacy_candidate.concept_id]
        freq = 1
        vector = spacy_kb.get_vector(scispacy_candidate.concept_id)
        spacy_candidate = Candidate(
            kb=spacy_kb,
            entity_hash=entity_hash,
            entity_freq=freq,
            entity_vector=vector,
            alias_hash=1,  # an integer is required here, but does not seem to be used anywhere
            prior_prob=0.0,  # a float is required here, but is not used when incl_prior is False on the EntityLinker
        )
        yield spacy_candidate


@spacy.registry.misc("scispacy_candidate_generator")
def scispacy_get_candidate_generator(
    scispacy_kb_name: str,
) -> Callable[[SpacyKnowledgeBase, Span], Iterable[Candidate]]:
    candidate_generator = CandidateGenerator(name=scispacy_kb_name)
    return functools.partial(scispacy_get_candidates, candidate_generator)
