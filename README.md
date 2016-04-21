
# UNET
UNET is a multi-core based, high performance machine learning framework, built on top of Spark,  supporting both data parallel and model parallel in massive scale.

Currently, UNET is based on Spark-2.0 (not released yet), but the source code is not made public yet. If you are interested, please contact me at zhazhan@gmail.com

#Running TestCases
    mvn package -DskipTests
    ./mnist.sh
    ./test.sh


## Examples
1st: We split the data into 3 splits, each one is trained by a model which is distributed to 2 machines. First, we train it with rbm (between 1st and 2nd layer), and then train it with denoising autoencoder (between 2nd and 3rd layer), and sgd is used to for the final supervised learning for classcification.

    val uc = new UNetContext(sc)
    val master = uc.getMaster("DBNNetSuite")
    val nnet: NNet = DBNNet.createDBNModel()
    val rawRDD = sc.textFile(
      "mnist")
    val trainRDD = rawRDD.map{ _.split(" ").map(_.toDouble)
    }.repartition(3)
    master.withNNet(nnet)
      .withPartitions(2)
      .withIterations(1000)
      .withInputRdd(Some(trainRDD))
    master.withExeStep(nnet.exeSteps("rbm-1")).train
    val r = master.withExeStep(nnet.exeSteps("rbm-1")).stack()
    master.withInputRdd(Some(r)).withExeStep(nnet.exeSteps("da")).train
    val r1 = master.withInputRdd(Some(r)).withExeStep(nnet.exeSteps("da")).stack()
    master.withInputRdd(Some(r1)).withExeStep(nnet.exeSteps("rbm-2")).train
    master.withInputRdd(Some(trainRDD)).withExeStep(nnet.exeSteps("sgd")).train
    master.shutDown
    
2nd:We split the data into 3 splits (data parallel), each one is trained by a model which is distributed to 3 machines (model parallel). First, we train it with Asynchronous SGD, then with Synchronous SGD, followed by QN and CG batching training method.

    val uc = new UNetContext(sc)
    val master = uc.getMaster("BatchQNConvNetSuite")
    val nnet: NNet = BatchQNConvNet.createConvModel()
    val rawRDD = sc.textFile(
      "mnist/data/mnist_train_input_combine")
    val trainRDD = rawRDD.map{ _.split(" ").map(_.toDouble)
    }.repartition(3)
    master.withSnapShot(0, -1)
      .withNNet(nnet)
      .withPartitions(3)
      .withMicroBatch(64)
      .withPipeLength(4)
      .withMicroPerIter(5)
      .withIterations(20000)
      .withInputRdd(Some(trainRDD))
      .withParams(Map(ModelParams.regularization -> "L1", ModelParams.snapshotIterNum -> "1",
      ModelParams.fetchStep -> "2", ModelParams.updateStep -> "2",
      ModelParams.pmGamma -> "1.0d", ModelParams.adaBound -> "true"))
    master.withExeStep(nnet.exeSteps.getStep("sgd")).train
    master
      .withParams(Map(ModelParams.regularization -> "None", ModelParams.paramInit -> "UNIFORM"))
      .withMicroBatch(64).withMicroPerIter(5)
      .withPipeLength(3).withPartitions(2).withExeStep(nnet.exeSteps.getStep("ssgd")).train
     master
       .withParams(Map(ModelParams.regularization -> "None", ModelParams.paramInit -> "UNIFORM"))
       .withMicroBatch(10000).withIterations(1000000).withPipeLength(1).withExeStep(nnet.exeSteps("qn")).train
     master
       .withParams(Map(ModelParams.regularization -> "None", ModelParams.paramInit -> "UNIFORM"))
       .withMicroBatch(10000).withIterations(1000000).withPipeLength(1).withExeStep(nnet.exeSteps("cg")).train
     master.shutDown
     
## Architecture

The system consists of three major components, Parameter Servers, Models, and Solver.

Each of them is orgnized as a RDD, and may consists of mulitple partitions. Note that there may be mulitple model Rdd.

Consider following scenario in training a model.

Data Parallel: Because the training data is huge, the system spawn 10 replicated models, each of them trains on one tenth of the data.
Model Parallel: Because the model is huge, each replicated models are distributed to 5 machines to overcome memory limitation and cpu bottleneck.
In such case, the system will have 1. One Parameter RDD: consists of 5 partitions, with each partition corresponds to one partitions of the model. 2. 10 Model RDDs: there are 10 replicated model rdd, with each of them consists of 5 partitions. 3. 1 Solver RDD to control the whole process of the training.

##DAG
Flexible network topology construction. Supporting each layer has multiple incoming/outgoing layers

#Algorithms
Denoising Auto-encoder, RBM, Stacking, Convolutional Neural Net, LSTM, Bidirectional-LSTM, Conjugate Gradient, L-BFGS, L1/L2/MaxNorm Regularization, DropOut, MaxPool, MeanPool, ReLU,  Momentum, Nesterov, and various Squashing functions.

##Parameter Server

The parameter server is the central repo for the system state. It is orgnized as a RDD with mulitple partitions. How the RDD is partitioned is determined by the partitions used in Model Parallel (discussed later).

It receives parameter update from sub-models, aggregates and feeds updated parameters to sub-models.

In synchronous mode (batch or syncSGD), parameter server receives command from solvers, and calucate the direction, e.g., in L-BGFS/Conjugate Gradient through vector operations, or calculate the gradient by averaging updated from sub-models.

In asynchrounous mode (asyncSGD), parameters receives request form sub-models and replies with updated parameters. The parameter algorithm is AdaGrad or other algorithms.

##Data Parallel
Training data are partitioned into multiple splits, and are trianed independently. Each split corresponds to one RDD in Spark, and calculates the sub-gradient synchronously in Batch or SyncSGD, or asynchronously in AsyncSGD. Correspondingly, it updates/fetch parameter server in sync/async manner.

##Model Parallel
Each model (RDD) can be further partitioned into mulitple splits, and are trained thorugh optimized software pipeline. Each split corresponds to one partition of the given RDD. In this way, each partition hosts one part of the parameters in the whole model.

Note that parameter server follows exactly the same way as how the model is partitioned in model parallel. By this way, each partition in the model knows exactly which parameter server partition hosts its parameter, and then know exactly where to update and fetch its parameters.

##Function Split
Because one model instance is split into mulitple partitions, to save the memory and network traffic overhead, we use the function split to achieve O(n) complexity with both memory and traffic overhead.

## Communication Overhead
The complexity of communication is the number of computation nodes (neurons), instead of edges.

## Memory Overhead
The complexity is also the number of nodes (neurons).

##Network Latency
Software piple to overcome the network latency.

##Performance:
All the computation and communication in UNET is implemented as async with high concurrency.

##Pipeline
With the software pipeline, the framework is able to leverage full cluster computation power, and reduce the network latency impact to achieve high performance. The technique is esp. useful in batch training, e.g., L-BFGS, Conjugate Gradient, etc or large minibatch in SGD.


##Parameter Server
Consists of a cluster of executors, and each of them held one shard of parameters. On the one hand, support upload/download from/to models with high concurrency. On the other hand, it also provides a rich set of vector operations for batch training, for example, compute directions, vector operations. It is extremely easy to extend with more operations.

##Model
One model is split into multiple parts, and communicate with netty in asynchronous way. It uploads/downloads parameters to parameter servers periodically in SGD, or receives command from Solver to perform operations. 

The basic responsibility of a model is compute the gradient on its own data split. In SGD, it locally updates gradient and sync with parameter servers.

##Solver
For SGD, it mainly monitor the model training process.
For Batch, e.g., L-BFGS, CG, it aggregte the values from models, and request the parameter server to calculate the directions, and request models to perform line search.

##Easy of Use
The framework is built on top of spark, and used as a package with fully capacity from Spark, including Spark-SQL, MlLib, GraphX, etc.

##Extensibility
To be extend to support other algorithms, e.g., K-Means, Matrix factorization.




