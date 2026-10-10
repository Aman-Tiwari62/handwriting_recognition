import torch


def collate_fn(batch):

    images, labels, texts = zip(*batch)

    # ---------------------------------------------
    # Image padding
    # ---------------------------------------------

    max_width = max(
        image.shape[2]
        for image in images
    )

    batch_size = len(images)
    channels = images[0].shape[0]
    height = images[0].shape[1]

    batch_images = torch.ones(
        batch_size,
        channels,
        height,
        max_width
    )

    input_widths = []

    for i, image in enumerate(images):

        width = image.shape[2]

        batch_images[
            i,
            :,
            :,
            :width
        ] = image

        input_widths.append(width)

    # ---------------------------------------------
    # Labels
    # ---------------------------------------------

    label_lengths = torch.tensor(
        [len(label) for label in labels],
        dtype=torch.long
    )

    labels = torch.cat(labels)

    input_widths = torch.tensor(
        input_widths,
        dtype=torch.long
    )

    return (
        batch_images,
        labels,
        input_widths,
        label_lengths,
        texts
    )