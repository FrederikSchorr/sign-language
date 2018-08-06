"""
https://github.com/FrederikSchorr/sign-language

Time measurement utilities
"""


import time

class Timer():
    def __init__(self):
        self.fTotal = 0.0
        return

    def start(self):
        self.fStart = time.time()
        return

    def stop(self):
        fDelta = time.time() - self.fStart
        self.fTotal += fDelta
        print("Execution time: %3.2f sec" % fDelta)
        return fDelta

    def sum(self):
        fTotal = self.fTotal
        print("Total execution time: %3.2f sec" % fTotal)
        self.fTotal = 0.0
        return fTotal


def unittest():
    timer = Timer()

    timer.start()
    for i in range(100):
        print(i)
    timer.stop()

    timer.start()
    for i in range(20):
        print(i)
    timer.stop()

    timer.sum()

if __name__ == '__main__':
    unittest()