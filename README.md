## clust4j
A Java-based set of clustering algorithm implementations for __classification__. Built and tested under JDK 1.7.0_79

____
### Dependencies:
 - [Apache commons math](https://commons.apache.org/proper/commons-math/), for use of the `AbstractRealMatrix` and `FastMath` classes.
  - The `commons-math` dependency is included in [dep/](https://github.com/tgsmith61591/clust4j/tree/master/dep/commons-math-3-3.2) and is already in the `.classpath`
 - [Apache log4j](http://logging.apache.org/log4j/2.x/), for use of the Logger class
  - The `log4j` dependency is included in [dep/](https://github.com/tgsmith61591/clust4j/tree/master/dep/apache-log4j-1.2.17) and is already in the `.classpath`.
  - In any `BaseClustererPlanner` class, invoke `.setVerbose(true)` to enable logging. Default logging location is: `/tmp/clust4j-${USERNAME}/clust4jlogs/`


____
### Example data (for reproducability):
    final int k = 2;
    final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(new double[][] {
        new double[] {0.005,     0.182751,  0.1284},
        new double[] {3.65816,   0.29518,   2.123316},
        new double[] {4.1234,    0.2301,    1.8900002}
    });	

    /* Test data, where necessary */
    final Array2DRowRealMatrix test  = new Array2DRowRealMatrix(new double[][] {
        new double[] {0.01302,   0.0012,   0.06948},
        new double[] {3.01837,   2.2293,   3.94812}
    });
		
    final int[] trainLabels = new int[] {0, 1, 1};

    


### Currently implemented algorithms:
- Partitional algorithms:
  - [*k*-Nearest Neighbor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), a non-parametric, supervised clustering method used for classification. 

            KNN knn = new KNN(mat, test, trainLabels, new KNN.KNNPlanner(k).setScale(true));
            knn.train();
            final int[] results = knn.getPredictedLabels(); // [0,1]

  - [*k*-Means](https://en.wikipedia.org/wiki/K-means_clustering), an unsupervised clustering method that aims to partition *n* observations into *k* clusters in which each observation belongs to the cluster with the nearest mean (centroid), serving as a prototype of the cluster.

            KMeans km = new KMeans(mat, new KMeans.BaseKCentroidPlanner(k).setScale(true));
            km.train();
            // Returns either [1,0] or [0,1] depending on seed:
            final int[] results = km.getPredictedLabels();

  - [*k*-Medoids](https://en.wikipedia.org/wiki/K-medoids), an unsupervised clustering method that chooses datapoints as centers (medoids or exemplars) and works with an arbitrary matrix of distances between datapoints instead of using the Euclidean norm.

            KMedoids km = new KMedoids(mat, new KMedoidsPlanner(k).setScale(true));
            km.train();
            // Returns either [1,0] or [0,1] depending on seed:
            final int[] results = km.getPredictedLabels();

- Hierarchical algorithms:
  - [Agglomerative](https://en.wikipedia.org/wiki/Hierarchical_clustering), a "bottom up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy. Agglomerative clustering is __not__ computationally friendly in how it scales. The agglomerative clustering procedure performs at O(n<sup>2</sup>), but far outperforms its cousin, [Divisive Clustering](https://github.com/tgsmith61591/clust4j#future-implementations).

            AgglomerativeClusterer a = new AgglomerativeClusterer(mat, new BaseHierarchicalPlanner().setScale(true));
            a.train();
            // Print the tree, where 1 is the root:
            System.out.println(a); // Agglomerative clusterer: {1=<5, 2>, 2=<4, 3>, 3=null, 4=null, 5=null}

----
### Separability metrics
A number of separability metrics are available for use:

 - [Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance) __distance__ (L<sup>2</sup> norm)
 - [Manhattan](https://en.wikipedia.org/wiki/Taxicab_geometry) __distance__ (L<sup>1</sup> norm)
 - [Minkowski](https://en.wikipedia.org/wiki/Minkowski_distance) __distance__ (L<sup>P</sup> norm)
 - [Cosine](https://en.wikipedia.org/wiki/Cosine_similarity) __similarity__
 - [Kernel](https://en.wikipedia.org/wiki/Kernel_method) __similarity__ methods (descriptions by César Souza<sup>[[1]](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/)</sup>)
   - [ANOVA kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#anova), a radial basis function kernel, just as the Gaussian and Laplacian kernels. It is [said to perform well](http://www.nicta.com.au/research/research_publications?sq_content_src=%2BdXJsPWh0dHBzJTNBJTJGJTJGcHVibGljYXRpb25zLmluc2lkZS5uaWN0YS5jb20uYXUlMkZzZWFyY2glMkZmdWxsdGV4dCUzRmlkJTNEMjYxJmFsbD0x) in multidimensional regression problems (Hofmann, 2008).

     <p align="center"><img src="http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Csum_%7Bk=1%7D%5En%20%5Cexp%20(-%5Csigma%20(x%5Ek%20-%20y%5Ek)%5E2)%5Ed" alt="ANOVA Kernel"></p>


   - [Cauchy kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#cauchy), a long-tailed kernel and can be used to give long-range influence and sensitivity over the high dimension space.

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cfrac%7B1%7D%7B1%20+%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%5E2%7D%7B%5Csigma%5E2%7D%20%7D)


   - [Circular kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#circular), an example of an isotropic stationary kernel that is positive definite in R<sup>2</sup>.

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cfrac%7B2%7D%7B%5Cpi%7D%20%5Carccos%20(%20-%20%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D)%20-%20%5Cfrac%7B2%7D%7B%5Cpi%7D%20%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D%20%5Csqrt%7B1%20-%20%5Cleft(%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D%20%5Cright)%5E2%7D)


   - [Exponential kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#exponential); closely related to the Gaussian kernel, with only the square of the norm left out. It is also a radial basis function kernel.

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cexp%5Cleft(-%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%20%7D%7B2%5Csigma%5E2%7D%5Cright))


   - [Gaussian kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#gaussian), an example of radial basis function kernel. The adjustable parameter sigma plays a major role in the performance of the kernel, and should be carefully tuned to the problem at hand. If overestimated, the exponential will behave almost linearly and the higher-dimensional projection will start to lose its non-linear power. In the other hand, if underestimated, the function will lack regularization and the decision boundary will be highly sensitive to noise in training data.

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cexp%5Cleft(-%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%20%5E2%7D%7B2%5Csigma%5E2%7D%5Cright))


   - [Hyperbolic Tangent (sigmoid/tanh) kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#sigmoid), also known as the Sigmoid Kernel and as the Multilayer Perceptron (MLP) kernel. The Sigmoid Kernel comes from the Neural Networks field, where the bipolar sigmoid function is often used as an activation function for artificial neurons.

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Ctanh%20(%5Calpha%20x%5ET%20y%20+%20c))


   - [Laplacian kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#laplacian); completely equivalent to the exponential kernel, except for being less sensitive for changes in the sigma parameter. Being equivalent, it is also a radial basis function kernel.

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cexp%5Cleft(-%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%20%7D%7B%5Csigma%7D%5Cright))


   - [Linear kernel](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/#linear), the simplest kernel function. It is given by the inner product `<x,y>` plus an optional constant c.

     ![Image](http://latex.codecogs.com/gif.latex?k(x,%20y)%20=%20x%5ET%20y%20+%20c)


   - Log kernel

     ![Image](http://latex.codecogs.com/png.latex?k(x,y)%20=%20-%20log%20(%5ClVert%20x-y%20%5CrVert%20%5Ed%20+%201))


   - Multiquadric (and Inverse Multiquadric) kernel

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Csqrt%7B%5ClVert%20x-y%20%5CrVert%5E2%20+%20c%5E2%7D)


   - Polynomial kernel

     ![Image](http://latex.codecogs.com/gif.latex?k(x,%20y)%20=%20(%5Calpha%20x%5ET%20y%20+%20c)%5Ed)


   - Power kernel

     ![Image](http://latex.codecogs.com/png.latex?k(x,y)%20=%20-%20%5ClVert%20x-y%20%5CrVert%20%5Ed)


   - Radial Basis kernel

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%20%5Cexp%5Cleft(-%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%20%7D%7B%5Csigma%7D%5Cright))


   - Rational Quadratic kernel

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%201%20-%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%5E2%7D%7B%5ClVert%20x-y%20%5CrVert%5E2%20+%20c%7D)


   - Spherical kernel

     ![Image](http://latex.codecogs.com/png.latex?k(x,%20y)%20=%201%20-%20%5Cfrac%7B3%7D%7B2%7D%20%5Cfrac%7B%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D%20+%20%5Cfrac%7B1%7D%7B2%7D%20%5Cleft(%20%5Cfrac%7B%20%5ClVert%20x-y%20%5CrVert%7D%7B%5Csigma%7D%20%5Cright)%5E3)


   - Spline kernel

     ![Image](http://latex.codecogs.com/png.latex?k(x,y)%20=%20%5Cprod_%7Bi=1%7D%5Ed%201%20+%20x_i%20y_i%20+%20x_i%20y_i%20%5Cmin(x_i,%20y_i)%20-%20%5Cfrac%7Bx_i%20+%20y_i%7D%7B2%7D%20%5Cmin(x_i,y_i)%5E2%20+%20%5Cfrac%7B%5Cmin(x_i,y_i)%5E3%7D%7B3%7D)



Notice the differentiation between *similarity*-based and *distance*-based geometrically separable metrics. All the clustering algorithms are able to handle any metric implementing the `GeometricallySeparable` interface; if the method also implements `SimilarityMetric`, the algorithm will attempt to *maximize* similarity, else it will try to *minimize* distance.


###### When to use similarity metrics over distance metrics?
Various similarity metrics—kernel methods, in particular—allow the clustering algorithm to segment the data in Hilbert Space<sup>[4]</sup>, which can—assuming the proper kernel is selected—allow the algorithm to identify "complex," or non-(hyper)spherically shaped clusters:

![Image](http://www.ml.uni-saarland.de/code/pSpectralClustering/images/clusters_11b_notitle2.png)

To initialize any clusterer with a kernel as the `GeometricallySeparable` metric (example uses `GaussianKernel`):

    final Kernel kernel = new GaussianKernel();
    KMedoids km = new KMedoids(mat, new KMedoidsPlanner(k).setSep(kernel));


----
### Future implementations*:
- Density-based:
  - [DBSCAN](http://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf), a density-based clustering algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).
  - [MeanShift](https://en.wikipedia.org/wiki/Mean_shift), a non-parametric feature-space analysis technique for locating the maxima of a density function, a so-called mode-seeking algorithm.

- Hierarchical algorithms:
  - [Divisive](https://en.wikipedia.org/wiki/Hierarchical_clustering), a "top down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy. 

*__Update__ (Nov. 2015): as of now, there are no immediate plans to implement Divisive Clustering. The best estimates for [DIANA](http://www.unesco.org/webworld/idams/advguide/Chapt7_1_5.htm)'s (DIvisive ANAlysis) runtime is O(2<sup>n</sup>)<sup>[7]</sup>, as opposed to Agglomerative Clustering's O(n<sup>2</sup>). The only reason for implementing it would, thus, be out of the sake of completeness in the family of Hierarchical Clustering.



### Things to note:
 - The default `AbstractClusterer.BaseClustererPlanner.getScale()` returns `false`. This decision was made in an attempt to mitigate data transformations in instances where the analyst may not expect/desire them.  Note that [normalization *is* recommended](http://datascience.stackexchange.com/questions/6715/is-it-necessary-to-standardize-your-data-before-clustering?newreg=f574bddafe484441a7ba99d0d02b0069) prior to clustering and can be set in any algorithm's respective `Planner` class.  Example on a `KMeans` constructor:


        // With normalization
        new KMeans(mat, new KMeans.BaseKCentroidPlanner(k).setScale(true));
        // Without normalization
        new KMeans(mat, k);

 - By default, logging is disabled. This can be enabled by instance in any `BaseClustererPlanner` class by invoking `AbstractClusterer.BaseClustererPlanner.setVerbose(true)`, or it can be set globally:

        AbstractClusterer.DEF_VERBOSE = true;





----

##### References:
 1. Souza C.R., [Kernel Functions for Machine Learning](http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/)
 2. Yu K., Ji L., Zhang X., [Kernel Nearest-Neighbor Algorithm](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.125.3253&rep=rep1&type=pdf)
 3. Ester M., Kriegel H.-P., Sander J.S., Xu X., 1996. [A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise](http://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf), Institute for Computer Science, University of Munich
 4. Chitta, R., [Kernel Clustering](http://www.cse.msu.edu/~cse902/S14/ppt/kernelClustering.pdf)
 5. [kernlab](https://github.com/cran/kernlab/blob/master/R/kernels.R) R package
 6. [h2o](https://github.com/h2oai/h2o-2) (for log wrapper structure)
 7. [Divisive Clustering](http://www.unesco.org/webworld/idams/advguide/Chapt7_1_5.htm)
