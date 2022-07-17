import spacy

from scispacy.linking import EntityLinker


def main():
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe(
        "scispacy_linker",
        config={"resolve_abbreviations": False, "linker_name": "umls"},
    )


if __name__ == "__main__":
    main()
