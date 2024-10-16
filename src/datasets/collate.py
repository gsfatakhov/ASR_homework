import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict], spec_size=128):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    # Извлекаем аудио, спектрограммы и закодированные тексты
    spectrograms = [torch.squeeze(item["spectrogram"]) for item in dataset_items]
    text_encoded = [torch.squeeze(item["text_encoded"]) for item in dataset_items]

    # Паддинг для спектрограмм
    padded_spectrograms = torch.stack(
        [F.pad(spec, (0, spec_size - spec.shape[1])) for spec in spectrograms]
    )

    # Паддинг для закодированного текста
    max_text_len = max([text.shape[0] for text in text_encoded])
    padded_texts = torch.stack(
        [F.pad(text, (0, max_text_len - text.shape[0])) for text in text_encoded]
    )

    # Найдите максимальную длину аудио
    max_audio_len = max([item["audio"].shape[1] for item in dataset_items])

    # Паддинг для всех аудиофайлов до максимальной длины
    padded_audio = torch.stack(
        [
            F.pad(item["audio"], (0, max_audio_len - item["audio"].shape[1]))
            for item in dataset_items
        ]
    )

    N = len(text_encoded)

    # Возвращаем батч
    result_batch = {
        "audio": padded_audio,
        "spectrogram": padded_spectrograms,
        "spectrogram_length": torch.tensor([spec_size] * N, dtype=torch.long),
        "text_encoded": padded_texts,
        "text_encoded_length": torch.tensor([max_text_len] * N, dtype=torch.long),
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
    return result_batch
