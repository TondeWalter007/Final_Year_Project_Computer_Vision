import numpy
import sklearn.metrics
import matplotlib.pyplot as plt


def AP_score():
    def precision_recall_curve(y_true, pred_scores, thresholds):
        precisions = []
        recalls = []

        for threshold in thresholds:
            y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]
            r = numpy.flip(sklearn.metrics.confusion_matrix(y_true, y_pred))

            precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
            recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")

            precisions.append(precision)
            recalls.append(recall)

        return precisions, recalls, r

    #y_true = ["positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
             # "positive",
              #"positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
             # "positive",
             # "positive"]

    #pred_scores = [0.97, 0.99, 0.99, 0.97, 0.97, 0.99, 1.0, 0.99, 1.0, 0.99, 1.0, 0.98, 1.0, 0.94, 0.99, 0.99, 1.0,
                  # 0.98,
                   #1.0]

    y_true = ["positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
              "positive",
              "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
              "positive",
              "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
              "positive",
              "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive"]

    pred_scores = [0.99, 0.95, 0.99, 0.99, 1.0, 0.98, 0.99, 0.99, 0.97, 0.99, 0.98, 1.0, 0.85, 0.98, 0.99, 0.96, 0.99,
                   0.0, 0.97, 0.96, 0.96, 0.91, 0.97, 1.0, 0.99, 0.99, 0.94, 1.0, 0.98, 0.98, 0.96, 0.99, 1.0, 1.0,
                   0.99]


    print(len(y_true))
    print(len(pred_scores))

    thresholds = numpy.arange(start=0.2, stop=0.7, step=0.05)

    precisions, recalls, r = precision_recall_curve(y_true=y_true,
                                                    pred_scores=pred_scores,
                                                    thresholds=thresholds)

    precisions.append(1)
    recalls.append(0)

    precisions = numpy.array(precisions)
    recalls = numpy.array(recalls)

    AP = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    print("Object Detection AP Score:", AP)


def precision_recall_graphs():
    # y_true = ["positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
    # "positive",
    # "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
    # "positive",
    # "positive"]

    # pred_scores = [0.97, 0.99, 0.99, 0.97, 0.97, 0.99, 1.0, 0.99, 1.0, 0.99, 1.0, 0.98, 1.0, 0.94, 0.99, 0.99, 1.0,
    # 0.98,
    # 1.0]

    y_true = ["positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
              "positive",
              "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
              "positive",
              "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive",
              "positive",
              "positive", "positive", "positive", "positive", "positive", "positive", "positive", "positive"]

    pred_scores = [0.99, 0.95, 0.99, 0.99, 1.0, 0.98, 0.99, 0.99, 0.97, 0.99, 0.98, 1.0, 0.85, 0.98, 0.99, 0.96, 0.99,
                   0.0, 0.97, 0.96, 0.96, 0.91, 0.97, 1.0, 0.99, 0.99, 0.94, 1.0, 0.98, 0.98, 0.96, 0.99, 1.0, 1.0,
                   0.99]

    print(len(pred_scores))

    threshold = 0.5
    y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]
    print(y_pred)

    r = numpy.flip(sklearn.metrics.confusion_matrix(y_true, y_pred))
    print(r)

    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
    print(precision)

    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
    print(recall)

    thresholds = numpy.arange(start=0.2, stop=0.7, step=0.05)
    print("Thresholds:", thresholds)

    def precision_recall_curve(y_true, pred_scores, thresholds):
        precisions = []
        recalls = []

        for threshold in thresholds:
            y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

            precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
            recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")

            precisions.append(precision)
            recalls.append(recall)

        return precisions, recalls

    precisions, recalls = precision_recall_curve(y_true=y_true,
                                                 pred_scores=pred_scores,
                                                 thresholds=thresholds)

    print("Precisions:", precisions)
    print("Recall:", recalls)

    plt.plot(recalls, precisions, linewidth=4, color="red")
    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.show()

    f1 = 2 * ((numpy.array(precisions) * numpy.array(recalls)) / (numpy.array(precisions) + numpy.array(recalls)))
    print("f1:", f1)

    plt.plot(recalls, precisions, linewidth=4, color="red", zorder=0)
    plt.scatter(recalls[5], precisions[5], zorder=1, linewidth=6)

    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.show()

    precisions = numpy.array(precisions)
    recalls = numpy.array(recalls)

    AP = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    print("Object Detection AP Score:", AP)


precision_recall_graphs()
AP_score()
