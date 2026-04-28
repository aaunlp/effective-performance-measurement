from datasets import Dataset
import spacy


nlp = spacy.load("en_core_web_sm", disable=["ner"])


def create_bio_tags(text, entities):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    bio_tags = ["O"] * len(tokens)

    for entity in entities:
        start_token = None
        end_token = None
        for i, token in enumerate(doc):
            if token.idx == entity["Start character"]:
                start_token = i
            if token.idx + len(token) == entity["End character"]:
                end_token = i

        if start_token is not None and end_token is not None:
            bio_tags[start_token] = f"{entity['Label']}"
            for i in range(start_token + 1, end_token + 1):
                pass

    return tokens, bio_tags


def process_dataset(data):
    processed_data = []
    for entry in data:
        tokens, labels = create_bio_tags(entry["text"], entry["entities"])
        processed_data.append({"tokens": tokens, "labels": labels})

    hf_dataset = Dataset.from_list(processed_data)

    return hf_dataset


def align_labels_with_tokens(labels, word_ids):
    label_all_tokens = False
    previous_word_id = None
    new_labels = []
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != previous_word_id:
            new_labels.append(labels[word_id])
        else:
            new_labels.append(labels[word_id] if label_all_tokens else -100)
        previous_word_id = word_id
    return new_labels


def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        max_length=512,
        truncation=True,
        is_split_into_words=True,
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
