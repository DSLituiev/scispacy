import argparse
from ast import AsyncFunctionDef
import spacy
import numpy as np

from spacy.kb import KnowledgeBase as SpacyKnowledgeBase
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from scispacy.linking_utils import (
    KnowledgeBase as ScispacyKnowledgeBase,
    DEFAULT_UMLS_PATH,
)


def main(kb_path: str, vocab_model_path: str, output_path: str, device: str):
    scispacy_kb = ScispacyKnowledgeBase(kb_path)
    nlp = spacy.load(vocab_model_path)

    spacy_kb = SpacyKnowledgeBase(vocab=nlp.vocab, entity_vector_length=768)

    description_encoder_tokenizer = AutoTokenizer.from_pretrained(
        "michiyasunaga/BioLinkBERT-base", model_max_length=512
    )
    description_encoder_model = AutoModel.from_pretrained(
        "michiyasunaga/BioLinkBERT-base"
    )
    description_encoder_model.to(device)

    entity_ids = []
    definitions = []
    for entity_id, entity in scispacy_kb.cui_to_entity.items():
        if entity.definition is None or entity.definition == "":
            continue

        entity_ids.append(entity_id)
        definitions.append(entity.definition)

    print(f"Encoding {len(entity_ids), len(definitions)} entities.")
    # output_entity_vectors = np.ndarray(shape=(len(entity_ids), 768))
    # output_freqs = [1 for _ in range(len(entity_ids))]
    batch_size = 8
    for i in tqdm(range(0, len(entity_ids), batch_size)):
        batch_entity_ids = entity_ids[i : i + batch_size]
        batch_definitions = definitions[i : i + batch_size]
        inputs = description_encoder_tokenizer(
            batch_definitions, padding=True, truncation=True, return_tensors="pt"
        )
        inputs.to(device)
        outputs = description_encoder_model(**inputs)
        last_hidden_states = outputs.last_hidden_state.detach().cpu().numpy()
        cls_encodings = last_hidden_states[:, 0, :]
        # output_entity_vectors[i : i + batch_size, :] = cls_encodings

        for j in range(len(batch_entity_ids)):
            entity_id = batch_entity_ids[j]
            entity_vector = cls_encodings[j, :]
            spacy_kb.add_entity(entity=entity_id, entity_vector=entity_vector, freq=1)

    # spacy_kb.set_entities(
    #     entity_list=entity_ids,
    #     freq_list=output_freqs,
    #     vector_list=output_entity_vectors,
    # )

    print(f"Entities in the KB: {len(spacy_kb.get_entity_strings())}")
    spacy_kb.to_disk(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kb_path",
        help="Path to the scispacy KB file.",
        type=str,
        default=DEFAULT_UMLS_PATH,
    )
    parser.add_argument(
        "--vocab_model_path",
        help="Path to the model whose vocab the KB will use.",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        help="Path to the output directory.",
        type=str,
        default="kb_output",
    )
    parser.add_argument("--device", help="Device to use.", type=str, default="cpu")

    args = parser.parse_args()
    main(args.kb_path, args.vocab_model_path, args.output_path, args.device)
