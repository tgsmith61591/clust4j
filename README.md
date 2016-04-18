[![Build status](https://travis-ci.org/tgsmith61591/clust4j.svg?branch=master)](https://travis-ci.org/tgsmith61591/clust4j) 
[![Code coverage](https://img.shields.io/codecov/c/github/tgsmith61591/clust4j/master.svg)](https://codecov.io/github/tgsmith61591/clust4j?branch=master)
![Supported JVM](https://img.shields.io/badge/java-1.7-blue.svg) 
![Supported JVM](https://img.shields.io/badge/java-1.8-blue.svg)
[![License](http://img.shields.io/:license-apache-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)

## clust4j
A Java-based set of __classification__ clustering algorithms. Built under JDK 1.7 and tested up to JDK 1.8. *This project is currently in ongoing development for the time being and should not be used in production environments*

___
### Installation:
__Clust4j 1.1.6 prerelease is now available under [releases](https://github.com/tgsmith61591/clust4j/releases)__. To build the bleeding edge, use the included gradle wrapper:

```bash
git clone https://github.com/tgsmith61591/clust4j.git
cd clust4j
./gradlew build ## Go grab some coffee... this takes a minute
## jar is at build/libs/clust4j-x.x.x.jar and includes dependencies
```

___
### Clust4j includes some datasets:
- Iris
- Wine
- Breast cancer

Usage:

```java
DataSet data;
data = ExampleDataSets.loadIris();
data = ExampleDataSets.loadWine();
data = ExampleDataSets.loadBreastCancer();
```


____
### Simple example data (to use for reproducability):

```java
final int k = 2;
final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(new double[][] {
    new double[] {0.005,     0.182751,  0.1284},
    new double[] {3.65816,   0.29518,   2.123316},
    new double[] {4.1234,    0.2301,    1.8900002}
});	
```

    


### Currently implemented algorithms:
- **Partitional algorithms**:
  - [*k*-Means](https://en.wikipedia.org/wiki/K-means_clustering), an unsupervised clustering method that aims to partition *n* observations into *k* clusters in which each observation belongs to the cluster with the nearest mean (centroid), serving as a prototype of the cluster.

        ```java
        KMeans km = new KMeansParameters(k).fitNewModel(mat);
        final int[] results = km.getLabels();
        ```

  - [*k*-Medoids](https://en.wikipedia.org/wiki/K-medoids), an unsupervised clustering method that chooses datapoints as centers (medoids or exemplars) and works with an arbitrary matrix of distances between datapoints instead of using the Euclidean norm.

        ```java
        KMedoids km = new KMedoidsParameters(k).fitNewModel(mat);
        final int[] results = km.getLabels();
        ```

  - [Affinity Propagation](https://en.wikipedia.org/wiki/Affinity_propagation), a clustering algorithm based on the concept of "message passing" between data points.Like `KMedoids`, Affinity Propagation finds "exemplars", members of the input set that are representative of clusters.

        ```java
        AffinityPropagation ap = new AffinityPropagationParameters().fitNewModel(mat);
        final int[] results = ap.getLabels();
        ```


- **Hierarchical algorithms**:
  - [HierarchicalAgglomerative](https://en.wikipedia.org/wiki/Hierarchical_clustering), a "bottom up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy. Agglomerative clustering is __not__ computationally friendly in how it scales. The agglomerative clustering procedure performs at O(n<sup>2</sup>), but far outperforms its cousin, [Divisive Clustering](http://nlp.stanford.edu/IR-book/html/htmledition/divisive-clustering-1.html).

        ```java
        HierarchicalAgglomerative a = new HierarchicalAgglomerativeParameters().fitNewModel(mat);
        final int[] results = a.getLabels();
        ```

- **Density-based algorithms**:
  - [DBSCAN](http://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf), a density-based clustering algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

        ```java
        DBSCAN db = new DBSCANParameters(0.75).fitNewModel(mat);
        final int[] results = db.getLabels();
        ```

  - [HDBSCAN](http://link.springer.com/chapter/10.1007%2F978-3-642-37456-2_14), a density-based clustering algorithm: performs DBSCAN over varying epsilon values and integrates the result to find a clustering that gives the best stability over epsilon. This implementation includes five algorithms: PrimsKD, PrimsBall, BoruvkaKD and BoruvkaBall.

        ```java
        HDBSCAN hdb = new HDBSCANParameters().fitNewModel(mat);
        final int[] results = hdb.getLabels();
        ```

  - [MeanShift](https://en.wikipedia.org/wiki/Mean_shift), a non-parametric feature-space analysis technique for locating the maxima of a density function, a so-called mode-seeking algorithm.

        ```java
        MeanShift ms = new MeanShiftParameters(0.5).fitNewModel(mat);
        final int[] results = ms.getLabels();
        ```

- **Neighbor point clustering algorithms**:
  - `NearestNeighbors`, a neighbor clusterer that will fit the *k*-nearest points for each record in a matrix.

        ```java
        NearestNeighbors nn = new NearestNeighborsParameters(k).fitNewModel(mat);
        Neighborhood neighborhood = nn.getNeighbors();
        int[][] indices = neighborhood.getIndices(); // The indices in order of nearness
        double[][] distances = neighborhood.getDistances(); // The corresponding distances
        ```

  - `RadiusNeighbors`, a neighbor clusterer that will fit the nearest points within a given radius.

        ```java
        RadiusNeighbors rn = new RadiusNeighborsParameters(1.5).fitNewModel(mat);
        Neighborhood neighborhood = rn.getNeighbors();
        int[][] indices = neighborhood.getIndices(); // The indices in order of nearness
        double[][] distances = neighborhood.getDistances(); // The corresponding distances
        ```

- **Supervised clustering algorithms**:
  - `NearestCentroid`, a supervised algorithm that fits centroids based on a set of observed labels.

        ```java
        NearestCentroid nc = new NearestCentroidParameters().fitNewModel(mat, new int[]{0,1,1});
        // you can use .predict(AbstractRealMatrix) to retrieve predicted class labels on new data
        ```



### Evaluating performance
All clustering algorithms that implement `Classifier` can also be scored. Supervised and unsupervised methods are scored in different manners. If we want to score the `NearestCentroid` model we fit above:

```java
// implicitly uses ground truth and predicted labels
double accuracy = nc.score();
```

Conversely, if we'd like to score an unsupervised algorithm, we have a few options. Every `UnsupervisedClassifier` implements a [`silhouetteScore`](https://en.wikipedia.org/wiki/Silhouette_%28clustering%29) method, but also implements a home-grown scoring algorithm called `UnsupervisedIndexAffinity`. This operates based on the following complication:

```java
// our k-means model predicts labels like this:
km.getLabels(); // {0,1,1}
// ...but the ground truth may actually be {9,15,15}
```

... this method, then, is an attempt to measure accuracy not traditionally, but by accounting for predicted segmentation in regards to actual label segmentation. It works by penalizing indices which are inappropriately associated with incorrect neighbor indices. In this regard a ground truth set of {0,1,0,2,2} and predicted set of {2,0,2,1,1} would be 100% accurate. Use as follows:

```java
km.indexAffinityScore(new int[]{50,10,10}); // 100%
```

Alternatively, the `silhouetteScore` method can be called as follows:

```java
km.silhouetteScore(); // Uses internally predicted labels
```



### Making predictions
Any algorithm implementing `BaseClassifier` or its sub-interfaces will implement the method `predict(AbstractRealMatrix)`. After the model has been fit, predictions can be made as follows (using `KMeans` for example):

```java
KMeans model = new KMeansParameters(2).fitNewModel(mat); // fit the model on the training data
Array2DRowRealMatrix newData = new Array2DRowRealMatrix(new double[][]{ // here's the new data
    new double[]{0.0, 0.0, 0.0},
    new double[]{3.0, 0.0, 2.0}
});
int[] predicted_labels = model.predict(newData); // returns {0, 1}
```

__Note__ that `HDBSCAN` does not currently implement a functional `predict` method, and will throw an `UnsupportedOperationException`.


----
### Separability metrics
A number of separability metrics are available for use:
 - [Bray-Curtis](https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity) distance
 - [Canberra](https://en.wikipedia.org/wiki/Canberra_distance) distance
 - [Chebyshev](https://en.wikipedia.org/wiki/Chebyshev_distance) distance
 - [Cosine](https://en.wikipedia.org/wiki/Cosine_similarity) __similarity__
 - [Dice](https://reference.wolfram.com/language/ref/DiceDissimilarity.html) distance
 - [Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance) distance (L<sup>2</sup> norm)
 - [Kernel](https://en.wikipedia.org/wiki/Kernel_method) __similarity__ methods (descriptions by César Souza<sup>[[1](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/)]</sup>)]
 - [Haversine](https://en.wikipedia.org/wiki/Haversine_formula) distance (for geospatial cluster analysis)
 - [Hamming](https://en.wikipedia.org/wiki/Hamming_distance) distance
 - [Kulsinsky](http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.spatial.distance.kulsinski.html) distance
 - [Manhattan](https://en.wikipedia.org/wiki/Taxicab_geometry) distance (L<sup>1</sup> norm)
 - [Minkowski](https://en.wikipedia.org/wiki/Minkowski_distance) distance (L<sup>P</sup> norm)
 - [Rogers-Tanimoto](https://reference.wolfram.com/language/ref/RogersTanimotoDissimilarity.html) distance  
 - [Russell-Rao](https://reference.wolfram.com/language/ref/RussellRaoDissimilarity.html) distance 
 - [Sokal-Sneath](https://reference.wolfram.com/language/ref/SokalSneathDissimilarity.html) distance
 - [Yule](https://reference.wolfram.com/language/ref/YuleDissimilarity.html) distance



----
### Kernel functions:
   - [ANOVA kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#anova), a radial basis function kernel, just as the Gaussian and Laplacian kernels. It is [said to perform well](http://www.nicta.com.au/research/research_publications?sq_content_src=%2BdXJsPWh0dHBzJTNBJTJGJTJGcHVibGljYXRpb25zLmluc2lkZS5uaWN0YS5jb20uYXUlMkZzZWFyY2glMkZmdWxsdGV4dCUzRmlkJTNEMjYxJmFsbD0x) in multidimensional regression problems (Hofmann, 2008).

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Csum_%7Bk=1%7D%5En%20%5Cexp%20(-%5Csigma%20(x%5Ek%20-%20y%5Ek)%5E2)%5Ed" alt="ANOVA Kernel"></p>


   - [Cauchy kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#cauchy), a long-tailed kernel and can be used to give long-range influence and sensitivity over the high dimension space.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cfrac%7B1%7D%7B1%20+%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%5E2%7D%7B%5Csigma%5E2%7D%20%7D" alt="Cauchy kernel"></p>


   - [Circular kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#circular), an example of an isotropic stationary kernel that is positive definite in R<sup>2</sup>.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cfrac%7B2%7D%7B%5Cpi%7D%20%5Carccos%20(%20-%20%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D)%20-%20%5Cfrac%7B2%7D%7B%5Cpi%7D%20%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D%20%5Csqrt%7B1%20-%20%5Cleft(%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D%20%5Cright)%5E2%7D" alt="Circular kernel"></p>


   - [Exponential kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#exponential); closely related to the Gaussian kernel, with only the square of the norm left out. It is also a radial basis function kernel.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cexp%5Cleft(-%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%20%7D%7B2%5Csigma%5E2%7D%5Cright)" alt="Exponential kernel"></p>


   - [Gaussian kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#gaussian), an example of radial basis function kernel. The adjustable parameter sigma plays a major role in the performance of the kernel, and should be carefully tuned to the problem at hand. If overestimated, the exponential will behave almost linearly and the higher-dimensional projection will start to lose its non-linear power. In the other hand, if underestimated, the function will lack regularization and the decision boundary will be highly sensitive to noise in training data.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cexp%5Cleft(-%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%20%5E2%7D%7B2%5Csigma%5E2%7D%5Cright)" alt="Gaussian kernel"></p>


   - [Hyperbolic Tangent (sigmoid/tanh) kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#sigmoid), also known as the Sigmoid Kernel and as the Multilayer Perceptron (MLP) kernel. The Sigmoid Kernel comes from the Neural Networks field, where the bipolar sigmoid function is often used as an activation function for artificial neurons.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Ctanh%20(%5Calpha%20x%5ET%20y%20+%20c)" alt="Sigmoid kernel"></p>


   - [Laplacian kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#laplacian); completely equivalent to the exponential kernel, except for being less sensitive for changes in the sigma parameter. Being equivalent, it is also a radial basis function kernel.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cexp%5Cleft(-%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%20%7D%7B%5Csigma%7D%5Cright)" alt="Laplacian kernel"></p>


   - [Linear kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#linear), the simplest kernel function. It is given by the inner product `<x,y>` plus an optional constant c.

<p align="center"><img src="http://latex.codecogs.com/gif.latex?k(x,%20y)%20=%20x%5ET%20y%20+%20c" alt="Linear kernel"></p>


   - [Log kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#log); seems to be particularly interesting for images, but is only conditionally positive definite.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,y)%20=%20-%20log%20(%5ClVert%20x-y%20%5CrVert%20%5Ed%20+%201)" alt="Log kernel"></p>


   - [Multiquadric](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#multiquadric) (and [Inverse Multiquadric](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#inverse_multiquadric)) kernel; can be used in the same situations as the Rational Quadratic kernel. In the case of the Inverse Multiquadric Kernel, it results in a kernel matrix with full rank [(Micchelli, 1986)](http://www.springerlink.com/content/w62233k766460945/) and thus forms a infinite dimension feature space.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Csqrt%7B%5ClVert%20x-y%20%5CrVert%5E2%20+%20c%5E2%7D" alt="Quadric kernels"></p>


   - [Polynomial kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#polynomial), a non-stationary kernel. Polynomial kernels are well suited for problems where all the training data is normalized.

<p align="center"><img src="http://latex.codecogs.com/gif.latex?k(x,%20y)%20=%20(%5Calpha%20x%5ET%20y%20+%20c)%5Ed" alt="Polynomial kernel"></p>


   - [Power kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#power), also known as the (unrectified) triangular kernel. It is an example of scale-invariant kernel [(Sahbi and Fleuret, 2004)](http://hal.archives-ouvertes.fr/docs/00/07/19/84/PDF/RR-4601.pdf) and is also only conditionally positive definite.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,y)%20=%20-%20%5ClVert%20x-y%20%5CrVert%20%5Ed" alt="Power kernel"></p>


   - [Radial Basis kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel), commonly used in support vector machine classification, the Radial Basis Function Kernel (RBF) may be recognized as the squared Euclidean distance between the two feature vectors where sigma is a free parameter. The feature space of the kernel has an infinite number of dimensions.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cexp%5Cleft(-%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%20%7D%7B%5Csigma%7D%5Cright)" alt="Radial basis kernel"></p>


   - [Rational Quadratic kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#rational); less computationally intensive than the Gaussian kernel and can be used as an alternative when using the Gaussian becomes too expensive.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%201%20-%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%5E2%7D%7B%5ClVert%20x-y%20%5CrVert%5E2%20+%20c%7D" alt="Rational quadratic kernel"></p>


   - [Spherical kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#spherical), similar to the circular kernel, but is positive definite in R<sup>3</sup>.

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%201%20-%20%5Cfrac%7B3%7D%7B2%7D%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D%20+%20%5Cfrac%7B1%7D%7B2%7D%20%5Cleft(%20%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D%20%5Cright)%5E3" alt="Spherical kernel"></p>


   - [Spline kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#spline), given as a piece-wise cubic polynomial, as derived in the works by [Gunn (1998)](http://www.svms.org/tutorials/Gunn1998.pdf).

<p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,y)%20=%20%5Cprod_%7Bi=1%7D%5Ed%201%20+%20x_i%20y_i%20+%20x_i%20y_i%20%5Cmin(x_i,%20y_i)%20-%20%5Cfrac%7Bx_i%20+%20y_i%7D%7B2%7D%20%5Cmin(x_i,y_i)%5E2%20+%20%5Cfrac%7B%5Cmin(x_i,y_i)%5E3%7D%7B3%7D" alt="Spline kernel"></p>



----
Notice the differentiation between *similarity*-based and *distance*-based geometrically separable metrics. All the clustering algorithms are able to handle any metric implementing the `GeometricallySeparable` interface.  Since `SimilarityMetric`'s `getSimilarity(...)` method should return the negative of `getDistance(...)`, separability metrics implementing `SimilarityMetric` implicitly will attempt to *maximize* similarity (as all clustering algorithms will minimize distance).  Classes implementing `SimilarityMetric` should define the `getDistance(double[], double[])` method to return `-getSimilarity(double[], double[])`.


###### When to use similarity metrics over distance metrics?
Various similarity metrics—kernel methods, in particular—allow the clustering algorithm to segment the data in Hilbert Space<sup>[[4](http://www.cse.msu.edu/~cse902/S14/ppt/kernelClustering.pdf)]</sup>, which can—assuming the proper kernel is selected—allow the algorithm to identify "complex," or non-(hyper)spherically shaped clusters:

![Image](http://www.ml.uni-saarland.de/code/pSpectralClustering/images/clusters_11b_notitle2.png)

Whereas a distance metric in Euclidean space may struggle with oddly shaped clusters<sup>[[9](http://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means)]</sup>:

![Image](http://www.ml.uni-saarland.de/code/pSpectralClustering/images/clusters_17b_notitle2.png)

To initialize any clusterer with a kernel as the `GeometricallySeparable` metric (example uses `GaussianKernel`):

```java
final Kernel kernel = new GaussianKernel();
KMedoids km = new KMedoids(mat, new KMedoidsPlanner(k).setSep(kernel));
```

__Note:__ though similarity metrics *may* be used with any clustering algorithm, it is recommended that they *not* be used with [density-based](https://github.com/tgsmith61591/clust4j/blob/master/src/com/clust4j/algo/AbstractDensityClusterer.java) clustering algorithms, as they seek "neighborhoods" around points and similarity metrics such as kernels will not accurately describe a point's neighborhood.  Using a similarity metric with a density-based algorithm will cause a warning to be logged and the algorithm will fall back to the default separability metric (Euclidean distance).

----

### Utilities
- **Matrix imputation**:
  - Mean imputation
  - Median imputation
  - Bootstrap (basic or smooth) imputation
  - KNN Imputation

- **Pipeline**:
  - Construct a pipeline of `PreProcessor`s through which to push new data, resulting in a cluster fit:
        
        ```java
        // Unsupervised:
        final KMedoidsParameters planner = new KMedoidsParameters(2).setVerbose(true);
        // Use of varargs for the PreProcessors is supported
        final UnsupervisedPipeline<KMedoids> pipe = new UnsupervisedPipeline<KMedoids>(planner, Normalize.CENTER_SCALE /*, ... */);
        // Push data through preprocessing pipeline and fit model
        KMedoids km = pipe.fit(mat);
        
        // Supervised:
        final NearestCentroidParameters s_planner = new NearestCentroidParameters().setVerbose(true);
        final SupervisedPipeline<NearestCentroid> sup_pip = new SupervisedPipeline<NearestCentroid>(s_planner, Normalize.CENTER_SCALE);
        NearestCentroid model = sup_pip.fit(mat, new int[]{0,1,1});
        ```



----

### Things to note:
 - The default `BaseClustererParameters.getScale()` currently returns `false`. This decision was made in an attempt to mitigate data transformations in instances where the analyst may not expect/desire them.  Note that [normalization *is* recommended](http://datascience.stackexchange.com/questions/6715/is-it-necessary-to-standardize-your-data-before-clustering?newreg=f574bddafe484441a7ba99d0d02b0069) prior to clustering and can be set in any algorithm's respective `Planner` class.  Example on a `KMeans` constructor:

    ```java
    // For normalization, simply add `.setScale(true)` on any `BaseClustererParameters` class
    new KMeansParameters(k).setScale(true);
    ```

 - By default, logging is disabled. This can be enabled by instance in any `BaseClustererParameters` class by invoking `.setVerbose(true)`.
 - Note that both of the above settings may be set globally:

    ```java
    AbstractClusterer.DEF_VERBOSE = true;
    AbstractClusterer.DEF_SCALE = true;
    ```



----

##### References:
 1. Souza C.R., [Kernel Functions for Machine Learning](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/)
 2. Yu K., Ji L., Zhang X., [Kernel Nearest-Neighbor Algorithm](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.125.3253&rep=rep1&type=pdf)
 3. Ester M., Kriegel H.-P., Sander J.S., Xu X., 1996. [A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise](http://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf), Institute for Computer Science, University of Munich
 4. Chitta, R., [Kernel Clustering](http://www.cse.msu.edu/~cse902/S14/ppt/kernelClustering.pdf)
 5. [kernlab](https://github.com/cran/kernlab/blob/master/R/kernels.R) R package
 6. [h2o](https://github.com/h2oai/h2o-2) (for log wrapper structure)
 7. [Divisive Clustering](http://www.unesco.org/webworld/idams/advguide/Chapt7_1_5.htm)
 8. [sklearn](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/cluster) clustering repository
 9. [The Drawbacks of k-Means](http://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means)
 10. [hdbscan](https://github.com/lmcinnes/hdbscan) python implementation
 11. [HDBSCAN](http://link.springer.com/chapter/10.1007%2F978-3-642-37456-2_14) research paper
