# from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool


def fib(n, m=4):
    print(m)
    # if n == 5:
    #     raise Exception("test error")
    if n == 1 or n == 0:
        return n
    else:
        return fib(n - 1, m) + fib(n - 2, m)


def f(x):
    return x * x


if __name__ == "__main__":
    print(fib(6, 2))
    with Pool(5) as p:
        print(p.starmap(fib, [(35, 5), (35,), (36, 1)]))
    # print(p.starmap(fib, [(35, 5), 35, (35, 1)]))
    # p = Pool(5)
    # print(p.starmap(fib, [(5, 5), (3,), (2, 1)]))
    print("done")
