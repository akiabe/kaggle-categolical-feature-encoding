def accuracy(y_true, y_pred):
    correct_counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1
    return correct_counter / len(y_true)

def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn

def accuracy_v2(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    accuracy_score = (tp + tn) / (tp + tn + fp + fn)
    return accuracy_score

def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    precision = tp / (tp + fp)
    return precision

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    score = 2 * p * r / (p + r)
    return score

def tpr(y_true, y_pred):
    return recall(y_true, y_pred)

def fpr(y_true, y_pred):
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return fp / (tn + fp)

if __name__ == "__main__":
    l1 = [0,1,1,1,0,0,0,1]
    l2 = [0,1,0,1,0,1,0,0]
    print(accuracy(l1, l2))
    print(true_positive(l1, l2))
    print(true_negative(l1, l2))
    print(false_positive(l1, l2))
    print(false_negative(l1, l2))
    print(accuracy_v2(l1, l2))
    print(precision(l1, l2))
    print(recall(l1, l2))
    print(f1(l1, l2))

    from sklearn import metrics
    print(metrics.accuracy_score(l1, l2))
    print(metrics.f1_score(l1, l2))

    tpr_list = []
    fpr_list = []

    y_true = [0, 0, 0, 0, 1, 0, 1,
              0, 0, 1, 0, 1, 0, 0, 1]
    y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
              0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
              0.85, 0.15, 0.99]
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

    for thresh in thresholds:
        temp_pred = [1 if x >= thresh else 0 for x in y_pred]
        temp_tpr = tpr(y_true, temp_pred)
        temp_fpr = fpr(y_true, temp_pred)
        tpr_list.append(temp_tpr)
        fpr_list.append(temp_fpr)

    import matplotlib.pyplot as plt
    #%matplotlib inline

    plt.figure(figsize=(7,7))
    plt.fill_between(fpr_list, tpr_list, alpha=0.4)
    plt.plot(fpr_list, tpr_list, lw=3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    plt.show()

    print(metrics.roc_auc_score(y_true, y_pred))