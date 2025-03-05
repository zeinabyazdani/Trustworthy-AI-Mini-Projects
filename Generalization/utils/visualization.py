
import random
import matplotlib.pyplot as plt


def data_visualization(batch_data, figsize = (3,3)):
    batch_image, lbl = batch_data
    batch_size = batch_image.shape[0]

    random_batch_index = random.randint(0, batch_size-1)
    random_image, random_lbl = batch_image[random_batch_index], lbl[random_batch_index]
    
    image_transposed = random_image.detach().numpy().transpose((1, 2, 0))
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image_transposed)
    ax.set_title(f"lbl:{random_lbl}")
    ax.set_axis_off()
