class Utils:
    @staticmethod
    def split_dataset(dataset):
        x = dataset.data
        y = dataset.target
        n_train, n_test = (1000, 100)
        x_train, y_train = x[:n_train, :], y[:n_train]
        x_test, y_test = x[n_train:n_train + n_test, :], y[n_train:n_train + n_test]
        return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
