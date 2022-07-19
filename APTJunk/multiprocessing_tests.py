# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool, Process, Queue
import multiprocessing as mp

# from multiprocessing.pool import ThreadPool as Pool
import time

import traceback


class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)

    def run(self):
        try:
            mp.Process.run(self)
        except Exception as e:
            print()
            print("failed")
            print(type(self))
            raise e
            # self.close()
            # self.terminate()
        finally:
            print("\n skip")


def fib(n, m=4):
    # if m == 0:
    #     raise Exception("test stop")
    # print(m)
    # if n == 5:
    #     raise Exception("test error")
    if n == 39:
        raise Exception("test error")
    if n == 1 or n == 0:
        return n
    else:
        return fib(n - 1, m) + fib(n - 2, m)


def child(*args):
    print("in child1")
    # print(args)
    p = Pool(len(args))
    res = []
    for i in args:
        res.append(p.apply_async(fib, (i,)))

    p.close()
    p.join()

    r1 = []
    for i, r in enumerate(res):
        r1.append(r.get())

    return r1


def child2(*args):
    print("in child2")
    res = []
    q = Queue()
    pool = []
    for arg in args:
        pool.append(Process(target=qu2, args=(q, arg)))
        pool[-1].start()
    for p in pool:
        p.join()
        p.close()
    while not q.empty():
        res.append(q.get())
    return res


def fib_print(*args):
    print(fib(*args))


def qu2(q, *args):
    return q.put(fib(*args))


def f(q, *args):
    return q.put(child2(*args))


if __name__ == "__main__":
    # t0 = time.monotonic()
    # print(fib(38, 2))
    # print(f"part 1 took {round(time.monotonic() - t0, 1)} seconds")
    t1 = time.monotonic()
    q = Queue()
    p1 = Process(target=f, args=(q, 38, 39, 38))
    p1.start()
    p1.join()
    p1.close()

    print(q.get())

    t2 = time.monotonic()
    print(f"done as {round(t2-t1, 2)} seconds")


# if __name__ == "__main__":
#     # t0 = time.monotonic()
#     # print(fib(38, 2))
#     # print(f"part 1 took {round(time.monotonic() - t0, 1)} seconds")
#     t1 = time.monotonic()
#     print("start")
#     processN = 4
#     p = Pool(processN)
#     numbers = [34, 35]
#     results = []
#     for number in numbers:
#         results.append(p.apply_async(child, (number, 6)))

#     # print(results[0].get())
#     print("right before close")
#     p.close()
#     print("right after close")
#     print("before join")
#     p.join()
#     print("right after join")
#     for i, r in enumerate(results):
#         print(f"\ni = {i}")
#         print(r)
#         try:
#             print(r.get() * -1)
#         except Exception as e:
#             print(e)
#     # print(p.starmap(fib, [(38, i) for i in range(processN)]))
#     # print(p.starmap(fib, [(35, 5), 35, (35, 1)]))
#     # p = Pool(5)
#     # print(p.starmap(fib, [(5, 5), (3,), (2, 1)]))
#     t2 = time.monotonic()
#     print(f"done as {round(t2-t1, 2)} seconds")


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


# if __name__ == "__main__":
#     # t0 = time.monotonic()
#     # print(fib(38, 2))
#     # print(f"part 1 took {round(time.monotonic() - t0, 1)} seconds")
#     t1 = time.monotonic()
#     print("start")
#     processN = 4
#     p = Pool(processN)
#     numbers = [34, 36]
#     results = []
#     for number in numbers:
#         results.append(p.apply_async(fib, (number, 6)))

#     print(results[0].get())
#     print("right before close")
#     p.close()
#     print("right after close")
#     print("before join")
#     p.join()
#     print("right after join")
#     for i, r in enumerate(results):
#         print(f"\ni = {i}")
#         print(r)
#         try:
#             print(r.get() * -1)
#         except Exception as e:
#             print(e)
#     # print(p.starmap(fib, [(38, i) for i in range(processN)]))
#     # print(p.starmap(fib, [(35, 5), 35, (35, 1)]))
#     # p = Pool(5)
#     # print(p.starmap(fib, [(5, 5), (3,), (2, 1)]))
#     t2 = time.monotonic()
#     print(f"done as {round(t2-t1, 2)} seconds")

# if __name__ == "__main__":
#     # t0 = time.monotonic()
#     # print(fib(38, 2))
#     # print(f"part 1 took {round(time.monotonic() - t0, 1)} seconds")
#     t1 = time.monotonic()
#     q = Queue(maxsize=2)
#     print(q.get())
#     # p1 = Process(target=f, args=(q, 3))
#     # # p2 = Process(target=f, args=(q, 5))
#     # p3 = Process(target=f, args=(q, 4))

#     # # p3 = Process(target=fib_print, args=(10,))
#     # p1.start()
#     # # p2.start()
#     # p3.start()

#     # # p3.start()
#     # print("a\n")
#     # print("b2\n")
#     # p1.join()
#     # print("c\n")
#     # # p2.join()
#     # p3.join()

#     # print("d\n")
#     # print(q.get())
#     # print("b1\n")
#     # print(q.get())
#     # print(q.get())
#     # p1.close()
#     # # p2.close()
#     # p3.close()

#     # print(f"length of q is {q.qsize()}")
#     t2 = time.monotonic()
#     print(f"done as {round(t2-t1, 2)} seconds")
