from pytimedinput import timedInput
import time

t1 = time.monotonic()
response, timedOut = timedInput("Hello", 5)
t2 = time.monotonic()

print(f"That took {round(t2-t1, 3)} seconds")
print(f"result = {response}")
