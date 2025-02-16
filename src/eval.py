

class BinaryConfusion:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert y_true.shape == y_pred.shape
        self.tp += torch.sum((y_true == 1) & (y_pred == 1))
        self.fn += torch.sum((y_true == 1) & (y_pred == 0))
        self.fp += torch.sum((y_true == 0) & (y_pred == 1))
        self.tn += torch.sum((y_true == 0) & (y_pred == 0))


    def get_f1(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def get_recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0

    def get_precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
    
    def get_iou(self):
        return self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) else 0