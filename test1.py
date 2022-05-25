import os

if not os.path.exists("fake_data"):
    print("test")

original_path = os.getcwd()
while not os.path.exists("data1/people"):
    os.chdir("..")
os.chdir("data1/people/jdtaylor")
if not os.path.exists("fake_data") and os.getcwd().split("/")[-1] == "jdtaylor":
    os.mkdir("fake_data")
os.chdir(original_path)
