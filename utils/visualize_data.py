import matplotlib.pyplot as plt


def display_training_data(data, sample_index):
    plt.figure(figsize=[5, 5])
    plt.subplot(121)
    plt.imshow(data[sample_index[0], :, :].squeeze(), cmap='gray')
    plt.title("Training Data: " + str(sample_index[0]))

    plt.subplot(122)
    plt.imshow(data[sample_index[1], :, :].squeeze(), cmap='gray')
    plt.title("Training Data: " + str(sample_index[1]))

    plt.show()


def display_training_results(train, metric):
    accuracy = train.history[metric + '_accuracy']
    val_accuracy = train.history['val_' + metric + '_accuracy']
    loss = train.history['loss']
    val_loss = train.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
