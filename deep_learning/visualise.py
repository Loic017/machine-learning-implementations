import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


def visualize_predictions_over_epochs(file_name, all_predictions, x_set):

    ###### NEED TO FIX
    # if len(all_predictions.shape) > 2:
    #     all_predictions = all_predictions.reshape(all_predictions.shape[0], -1)

    # if all_predictions.shape[-1] == 1:
    #     all_predictions = all_predictions.squeeze(axis=-1)

    # num_epochs = all_predictions.shape[0]

    # if len(x_set) != all_predictions.shape[1]:
    #     x_set = x_set[: all_predictions.shape[1]]

    fig, ax = plt.subplots()

    def update_plot(epoch):
        ax.clear()
        ax.scatter(x_set, all_predictions[epoch], color="blue")
        ax.set_title(f"Epoch {epoch + 1}")
        ax.set_xlabel("y_hat")
        ax.set_ylabel("X")
        ax.set_ylim(0, 1)

    ani = animation.FuncAnimation(fig, update_plot, frames=num_epochs, interval=500 / 4)

    writer = PillowWriter(fps=10)
    ani.save(file_name, writer=writer)

    plt.show()
