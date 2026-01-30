import matplotlib.pyplot as plt

def save_plots(train_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss grafigi
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-o', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy grafigi
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, 'b-s', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../2026_training_results.png')
    print("Grafiklar '../2026_training_results.png' sifatida saqlandi.")