import numpy;

dataSet = [
  [1, -890],
  [2, -1411],
  [2, -1560],
  [3, -2220],
  [3, -2091],
  [4, -2878],
  [5, -3537],
  [6, -3268],
  [6, -3920],
  [6, -4163],
  [8, -5471],
  [10, -5157]
];

def hFunction(originO0, originO1, x):
  return originO0 + originO1 * x;

def reduceO0(originO0, originO1, samples):
  temp = 0;
  for (x, y) in samples:
    temp += hFunction(originO0, originO1, x) - y;
  return temp / len(samples);

def reduceO1(originO0, originO1, samples):
  temp = 0;
  for (x, y) in samples:
    temp += (hFunction(originO0, originO1, x) - y) * x;
  return temp / len(samples);

# pylint: disable=W0622
def reduce(samples, originO0, originO1, step, count):
  for _ in range(0, count):
    tempO0 = originO0 - step * reduceO0(originO0, originO1, samples);
    tempO1 = originO1 - step * reduceO1(originO0, originO1, samples);
    originO0 = tempO0;
    originO1 = tempO1;
  return originO0, originO1


if __name__ == '__main__':
  a, b = reduce(dataSet, 0, 0, 0.01, 10000);
  print(a);
  print(b);
