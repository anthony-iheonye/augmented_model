from tensorflow.keras.callbacks import Callback


class ResetModelCounter(Callback):
    """
    A custom Keras callback that resets an internal counter in the model at the start of each epoch.

    This callback is useful when the model maintains a custom state (e.g., frame counter, step tracker,
    or class statistics) that should be reset at the beginning of every training epoch.

    The model using this callback must implement a `reset_counter()` method.

    Example:
        >>> model = MyCustomModel()
        >>> model.reset_counter = lambda: print("Counter reset!")
        >>> callback = ResetModelCounter()
        >>> model.fit(x_train, y_train, callbacks=[callback])

    Methods:
        on_epoch_begin(epoch, logs=None): Calls `self.model.reset_counter()` at the start of each epoch.
    """
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_counter()
