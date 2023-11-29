from matplotlib import pyplot
import regex
import numpy as np

with open("./training_output_losses.txt", "r") as file:
    text = file.read()

losses = regex.findall(r"(Test|Average) Loss: ([0-9.]*)", text)
trainings = []
tests = []

for match in losses:
    matchtype, loss_str = match
    if matchtype == "Average":
        trainings.append((len(trainings) + 1, float(loss_str)))
    elif matchtype == "Test":
        tests.append((len(trainings), float(loss_str)))
    else:
        print(match)
        quit()

trainxs, trainys = zip(*trainings)
testxs, testys = zip(*tests)
basexs, baseys = [0, len(trainings) + 1], [1.283, 1.283]

pyplot.plot(trainxs, trainys)
pyplot.plot(testxs, testys)
pyplot.plot(basexs, baseys)

pyplot.title("Loss by Epochs")
pyplot.xlabel("Epochs")
pyplot.ylabel("MSE loss")
pyplot.legend(["Training Loss", "Test Loss", "Baseline"])
print(testys)
pyplot.show()