# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
import multiprocessing as mp

# from multiprocessing.pool import ThreadPool as Pool
import time

import traceback


# class Process(mp.Process):
#     def __init__(self, *args, **kwargs):
#         mp.Process.__init__(self, *args, **kwargs)
#         self._pconn, self._cconn = mp.Pipe()
#         self._exception = None

#     def run(self):
#         try:
#             mp.Process.run(self)
#             self._cconn.send(None)
#         except Exception as e:
#             tb = traceback.format_exc()
#             self._cconn.send((e, tb))
#             # raise e  # You can still rise this exception if you need to

#     @property
#     def exception(self):
#         if self._pconn.poll():
#             self._exception = self._pconn.recv()
#         return self._exception


def fib(n, m=4):
    # if m == 0:
    #     raise Exception("test stop")
    # print(m)
    # if n == 5:
    #     raise Exception("test error")
    # if n == 36:
    #     raise Exception("test error")
    if n == 1 or n == 0:
        return n
    else:
        return fib(n - 1, m) + fib(n - 2, m)


def fib_print(*args):
    print(fib(*args))


def f(x):
    return x * x


# if __name__ == "__main__":
#     # t0 = time.monotonic()
#     # print(fib(38, 2))
#     # print(f"part 1 took {round(time.monotonic() - t0, 1)} seconds")
#     t1 = time.monotonic()
#     print("start")
#     processN = 4
#     p1 = Process(target=fib)
#     t2 = time.monotonic()
#     print(f"done as {round(t2-t1, 1)} seconds")


if __name__ == "__main__":
    # t0 = time.monotonic()
    # print(fib(38, 2))
    # print(f"part 1 took {round(time.monotonic() - t0, 1)} seconds")
    t1 = time.monotonic()
    print("start")
    processN = 4
    p = Pool(processN)
    numbers = [66, 66, 66, 66]
    results = []
    for number in numbers:
        results.append(p.apply_async(fib_print, (number, 6)))

    print("right before close")
    p.close()
    print("right after close")
    print("before join")
    p.join()
    print("right after join")
    for i, r in enumerate(results):
        print(f"i = {i}")
        print(r)
        try:
            r.get()
        except Exception as e:
            print(e)
    # print(p.starmap(fib, [(38, i) for i in range(processN)]))
    # print(p.starmap(fib, [(35, 5), 35, (35, 1)]))
    # p = Pool(5)
    # print(p.starmap(fib, [(5, 5), (3,), (2, 1)]))
    t2 = time.monotonic()
    print(f"done as {round(t2-t1, 1)} seconds")
