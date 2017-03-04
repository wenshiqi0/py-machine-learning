import matplotlib.pyplot as plt;
import numpy;

# simple for logistic [0, 1]
def sigmoid(param):
  return 1.0/(1.0 + numpy.exp(-param));

def loadDataSet(filePath):
  dataSet = [];
  classSet = [];
  with open(filePath) as file:
    for line in file.readlines():
      onelineArr = line.strip().split();
      dataSet.append([1.0, float(onelineArr[0]), float(onelineArr[1])]);
      classSet.append(int(onelineArr[2]));
  return dataSet, classSet;

def initRenderDataSet(dataSet, classSet):
  for i in range(0, len(dataSet) - 1):
    renderType = 'o';
    if (classSet[i] == 1):
      renderType = "+";
    plt.scatter(dataSet[i][1], dataSet[i][2], marker=renderType, color="black");

def renderClassification(weights):
  x = numpy.arange(-4.0, 4.0, 1);
  y = (- weights[0] - weights[1] * x)/weights[2];
  # weights is a matrix, so need to transform.
  y = numpy.array(y)[0];
  plt.plot(x, y);

def startRender():
  plt.title('Render');
  plt.legend();
  plt.grid(True);
  plt.show();

# Next: math methods under python
def gradAscent(dataSet, classSet):
  dataMat = numpy.mat(dataSet);
  classMat = numpy.mat(classSet).transpose();
  m, n = numpy.shape(dataMat);
  weigts = numpy.ones((n, 1));
  for i in range(500):
    h = sigmoid(dataMat * weigts);
    error = classMat - h;
    weigts = weigts + 0.001 * dataMat.transpose() * error;
  return weigts;                                                                                                                                                                                                                

if __name__ == '__main__':
  dataSet, classSet = loadDataSet('./logistic/dataSet.txt');
  initRenderDataSet(dataSet, classSet);
  weights = gradAscent(dataSet, classSet);
  renderClassification(weights);
  startRender();