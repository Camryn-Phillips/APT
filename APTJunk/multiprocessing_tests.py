# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool

# from multiprocessing.pool import ThreadPool as Pool
import time


def fib(n, m=4):
    # if m == 0:
    #     raise Exception("test stop")
    # print(m)
    # if n == 5:
    #     raise Exception("test error")
    if n == 1 or n == 0:
        return n
    else:
        return fib(n - 1, m) + fib(n - 2, m)


def f(x):
    return x * x


if __name__ == "__main__":
    # t0 = time.monotonic()
    # print(fib(38, 2))
    # print(f"part 1 took {round(time.monotonic() - t0, 1)} seconds")
    t1 = time.monotonic()
    print("start")
    processN = 4

    # with Pool(processN) as p:
    #     print(p.starmap(fib, [(38, i) for i in range(processN)]))
    # print(p.starmap(fib, [(35, 5), 35, (35, 1)]))
    # p = Pool(5)
    # print(p.starmap(fib, [(5, 5), (3,), (2, 1)]))
    t2 = time.monotonic()
    print(f"done as {round(t2-t1, 1)} seconds")
