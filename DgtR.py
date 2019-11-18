import numpy


def _search(y, num):
    arr = []
    for i, el in enumerate(y):
        if el == num:
            arr.append(i)
    return arr


class DigitRecignizer(object):
    def __init__(self):
        self.model = []

    def fit(self, x_train, y_train, param=0.8):
        arr = numpy.zeros(10, dtype=int)
        for i in range(10):
            i_arr = _search(y_train, i)
            for j in i_arr:
                count = 0
                for row in x_train[j]:
                    for col in row:
                        if col >= param:
                            count += 1
                arr[i] += count
            arr[i] /= len(i_arr)
        self.model = arr

    def predict(self, x_test):
        result = []
        for m in x_test:
            count = 0
            for row in m:
                for col in row:
                    if col >= 0.1:
                        count += 1
            result.append(count)

        pred = []
        for num in result:
            _max = 0
            index = 0
            for i, el in enumerate(self.model):
                if el <= num:
                    if num - el < 3:
                        index = i
                else:
                    index = i

            pred.append(index)
        print(pred)
        return pred
