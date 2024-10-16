import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict], n_feats=256):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
        n_feats (int): the width to pad the spectrograms to.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    spectrograms = [torch.squeeze(item["spectrogram"]) for item in dataset_items]
    text_encoded = [torch.squeeze(item["text_encoded"]) for item in dataset_items]

    max_spec_len = max([spec.shape[1] for spec in spectrograms])
    padded_spectrograms = torch.stack(
        [
            F.pad(spec, (0, max_spec_len - spec.shape[1], 0, n_feats - spec.shape[0]))
            for spec in spectrograms
        ]
    )

    max_text_len = max([text.shape[0] for text in text_encoded])
    padded_texts = torch.stack(
        [F.pad(text, (0, max_text_len - text.shape[0])) for text in text_encoded]
    )

    max_audio_len = max([item["audio"].shape[1] for item in dataset_items])

    padded_audio = torch.stack(
        [
            F.pad(item["audio"], (0, max_audio_len - item["audio"].shape[1]))
            for item in dataset_items
        ]
    )

    N = len(text_encoded)

    result_batch = {
        "audio": padded_audio,
        "spectrogram": padded_spectrograms,
        "spectrogram_length": torch.tensor(
            [spec.shape[1] for spec in spectrograms], dtype=torch.long
        ),
        "text_encoded": padded_texts,
        "text_encoded_length": torch.tensor([max_text_len] * N, dtype=torch.long),
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
    return result_batch
