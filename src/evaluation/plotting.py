import matplotlib.pyplot as plt


def plot_input_output_batch(batch_test_predictions, ci):

    for i, tuple_pred in enumerate(batch_test_predictions):
        input_signal_batch = tuple_pred[0]
        output_signal_batch = tuple_pred[1]
        ci_pred_batch = ci[i]
        list_index_ts = input_signal_batch['time_of_day'][:, 0, 0]
        ci_ts = ci_pred_batch[:, 0, 0, :]
        ts = output_signal_batch['targets'][:, 0, 0]
        plt.plot(list_index_ts, ts, color='b')
        pred_y = [i[1] for i in ci_ts]
        lower_ci = [i[0] for i in ci_ts]
        upper_ci = [i[2] for i in ci_ts]
        plt.fill_between(list_index_ts, lower_ci,
                         upper_ci, color='b', alpha=.1)
       
    plt.show()


def plot_time_series(list_ts, list_index_ts):
    for i, ts in enumerate(list_ts):
        plt.plot(list_index_ts[i], ts)
    plt.show()


def plot_time_series_pdf(ts, pdf_ts, list_index_ts):
    plt.plot(list_index_ts[0], ts)
    for sample in pdf_ts:

        plt.plot(list_index_ts[1], sample, 'o', markersize=1)
    plt.show()


def plot_time_series_ci(ts, ci_ts, list_index_ts):
    plt.plot(list_index_ts[0], ts)
    pred_y = [i[1] for i in ci_ts]
    lower_ci = [i[0] for i in ci_ts]
    upper_ci = [i[2] for i in ci_ts]
    plt.fill_between(list_index_ts[1], lower_ci, upper_ci, color='b', alpha=.1)
    plt.show()


def visualize_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):

    number_timestamp = 1/X_train[1, 0, 0, 1]
    list_ts = X_train[:, :, 0, 0].T.tolist()
    x = X_train[:, 0, 0, 1] * number_timestamp
    list_index_ts = [x for i in list_ts]

    for i, ts in enumerate(list_ts):
        if i > 0:
            break
        plt.plot(list_index_ts[i], ts, 'r')

    list_ts = X_valid[:, :, 0, 0].T.tolist()
    x = X_valid[:, 0, 0, 1] * number_timestamp
    list_index_ts = [x for i in list_ts]

    for i, ts in enumerate(list_ts):
        if i > 0:
            break
        plt.plot(list_index_ts[i], ts, 'g')

    list_ts = X_test[:, :, 0, 0].T.tolist()
    x = X_test[:, 0, 0, 1] * number_timestamp
    list_index_ts = [x for i in list_ts]

    for i, ts in enumerate(list_ts):
        if i > 0:
            break
        plt.plot(list_index_ts[i], ts, 'b')

    plt.show()
