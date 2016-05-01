/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.data;


abstract public class ExampleDataSets {
	
	/**
	 * 1. Title: Iris Plants Database Updated Sept 21 by C.Blake - Added
	 * discrepency information
	 * 
	 * <p>
	 * 2. Sources: 
	 * <ul>
	 * <li>(a) Creator: R.A. Fisher 
	 * <li>(b) Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov) (c) Date: July, 1988
	 * </ul>
	 * 
	 * <p>
	 * 3. Past Usage (far too many to exhaustively list): 
	 * <ul>
	 * <li>1. Fisher,R.A. "The use of multiple measurements in taxonomic problems"
	 * Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
	 * Mathematical Statistics" (John Wiley, NY, 1950). 
	 * <li>2. Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis. (Q327.D83)
	 * John Wiley & Sons. ISBN 0-471-22361-1. See page 218. 
	 * <li>3. Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System Structure and
	 * Classification Rule for Recognition in Partially Exposed Environments".
	 * IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol.
	 * PAMI-2, No. 1, 67-71. -- Results: -- very low misclassification rates (0%
	 * for the setosa class) 
	 * <li>4. Gates, G.W. (1972)
	 * "The Reduced Nearest Neighbor Rule". IEEE Transactions on Information
	 * Theory, May 1972, 431-433. -- Results: -- very low misclassification
	 * rates again 5. See also: 1988 MLC Proceedings, 54-64. Cheeseman et al's
	 * AUTOCLASS II conceptual clustering system finds 3 classes in the data.
	 * </ul>
	 * 
	 * 4. Relevant Information: --- This is perhaps the best known database to
	 * be found in the pattern recognition literature. Fisher's paper is a
	 * classic in the field and is referenced frequently to this day. (See Duda
	 * & Hart, for example.) The data set contains 3 classes of 50 instances
	 * each, where each class refers to a type of iris plant. One class is
	 * linearly separable from the other 2; the latter are NOT linearly
	 * separable from each other. --- Predicted attribute: class of iris plant.
	 * --- This is an exceedingly simple domain. --- This data differs from the
	 * data presented in Fishers article (identified by Steve Chadwick,
	 * spchadwick@espeedaz.net ) The 35th sample should be:
	 * 4.9,3.1,1.5,0.2,"Iris-setosa" where the error is in the fourth feature.
	 * The 38th sample: 4.9,3.6,1.4,0.1,"Iris-setosa" where the errors are in
	 * the second and third features.
	 * 
	 * <p>
	 * 5. Number of Instances: 150 (50 in each of three classes)
	 * 
	 * <p>
	 * 6. Number of Attributes: 4 numeric
	 * 
	 * <p>
	 * 7. Attribute Information: 
	 * <ul>
	 * <li>1. sepal length in cm 
	 * <li>2. sepal width in cm 
	 * <li>3. petal length in cm 
	 * <li>4. petal width in cm
	 * </ul>
	 * 
	 * 8. Missing Attribute Values: None
	 * 
	 * <p>
	 * 9. Class Distribution: 33.3% for each of 3 classes.
	 * 
	 * @return the iris dataset
	 * @see <a href="https://archive.ics.uci.edu/ml/datasets/Iris">ics.uci.edu</a>
	 */
	public final static DataSet loadIris() {
		return IrisLoader.load();
	}
	
	
	/**
	 * 1. Title of Database: Wine recognition data Updated Sept 21, 1998 by
	 * C.Blake : Added attribute information
	 *
	 * <p>
	 * 2. Sources: (a) Forina, M. et al, PARVUS - An Extendible Package for Data
	 * Exploration, Classification and Correlation. Institute of Pharmaceutical
	 * and Food Analysis and Technologies, Via Brigata Salerno, 16147 Genoa,
	 * Italy.
	 *
	 * <p>
	 * (b) Stefan Aeberhard, email: stefan@coral.cs.jcu.edu.au (c) July 1991 3.
	 * Past Usage:
	 * 
	 * <p>
	 * (1) S. Aeberhard, D. Coomans and O. de Vel, Comparison of Classifiers in
	 * High Dimensional Settings, Tech. Rep. no. 92-02, (1992), Dept. of
	 * Computer Science and Dept. of Mathematics and Statistics, James Cook
	 * University of North Queensland. (Also submitted to Technometrics).
	 * 
	 * <p>
	 * The data was used with many others for comparing various classifiers. The
	 * classes are separable, though only RDA has achieved 100% correct
	 * classification. (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1%
	 * (z-transformed data)) (All results using the leave-one-out technique)
	 * 
	 * <p>
	 * In a classification context, this is a well posed problem with
	 * "well behaved" class structures. A good data set for first testing of a
	 * new classifier, but not very challenging.
	 * 
	 * <p>
	 * (2) S. Aeberhard, D. Coomans and O. de Vel,
	 * "THE CLASSIFICATION PERFORMANCE OF RDA" Tech. Rep. no. 92-01, (1992),
	 * Dept. of Computer Science and Dept. of Mathematics and Statistics, James
	 * Cook University of North Queensland. (Also submitted to Journal of
	 * Chemometrics).
	 * 
	 * <p>
	 * Here, the data was used to illustrate the superior performance of the use
	 * of a new appreciation function with RDA.
	 * 
	 * <p>
	 * 4. Relevant Information:
	 * 
	 * <p>
	 * -- These data are the results of a chemical analysis of wines grown in
	 * the same region in Italy but derived from three different cultivars. The
	 * analysis determined the quantities of 13 constituents found in each of
	 * the three types of wines.
	 * 
	 * <p>
	 * -- I think that the initial data set had around 30 variables, but for
	 * some reason I only have the 13 dimensional version. I had a list of what
	 * the 30 or so variables were, but a.) I lost it, and b.), I would not know
	 * which 13 variables are included in the set.
	 * 
	 * <p>
	 * -- The attributes are (dontated by Riccardo Leardi,
	 * riclea@anchem.unige.it ) 
	 * 
	 * <ul>
	 * <li>1) Alcohol 
	 * <li>2) Malic acid 
	 * <li>3) Ash 
	 * <li>4) Alcalinity of ash 
	 * <li>5) Magnesium 
	 * <li>6) Total phenols 
	 * <li>7) Flavanoids 
	 * <li>8) Nonflavanoid phenols
	 * <li>9) Proanthocyanins 
	 * <li>10) Color intensity 
	 * <li>11) Hue 
	 * <li>12) OD280/OD315 of diluted wines 
	 * <li>13) Proline
	 * </ul>
	 * 
	 * <p>
	 * 5. Number of Instances:
	 * <ul>
	 * <li>class 0: 59 
	 * <li>class 2: 71 
	 * <li>class 3: 48
	 * </ul>
	 * 
	 * 6. Number of Attributes: 13
	 * 
	 * <p>
	 * 7. For Each Attribute:  All attributes are continuous
	 * 
	 * <p>
	 * 8. Missing Attribute Values: None
	 * 
	 * @return wine dataset
	 * @see <a href="https://archive.ics.uci.edu/ml/datasets/Wine">ics.uci.edu</a>
	 */
	public static DataSet loadWine() {
		return WineLoader.load();
	}
	
	
	/**
	 * 1. Title: Wisconsin Diagnostic Breast Cancer (WDBC)
	 * 
	 * <p>
	 * 2. Source Information
	 * 
	 * <ul>
	 * <li>a) Creators: Dr. William H. Wolberg, General Surgery Dept., University of Wisconsin,
	 * Clinical Sciences Center, Madison, WI 53792
	 * wolberg@eagle.surgery.wisc.edu
	 * 
	 * W. Nick Street, Computer Sciences Dept., University of Wisconsin, 1210
	 * West Dayton St., Madison, WI 53706 street@cs.wisc.edu 608-262-6619
	 * 
	 * Olvi L. Mangasarian, Computer Sciences Dept., University of Wisconsin,
	 * 1210 West Dayton St., Madison, WI 53706 olvi@cs.wisc.edu
	 * 
	 * <li>b) Donor: Nick Street
	 * 
	 * <li>c) Date: November 1995
	 * </ul>
	 * 
	 * 3. Past Usage:
	 * 
	 * <p>
	 * W.N. Street, W.H. Wolberg and O.L. Mangasarian Nuclear feature extraction
	 * for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on
	 * Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
	 * San Jose, CA, 1993.
	 * 
	 * <p>
	 * OR literature:
	 * <p>
	 * O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis
	 * and prognosis via linear programming. Operations Research, 43(4), pages
	 * 570-577, July-August 1995.
	 * <p>
	 * Medical literature:
	 * <p>
	 * W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning
	 * techniques to diagnose breast cancer from fine-needle aspirates. Cancer
	 * Letters 77 (1994) 163-171.
	 * <p>
	 * W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Image analysis and
	 * machine learning applied to breast cancer diagnosis and prognosis.
	 * Analytical and Quantitative Cytology and Histology, Vol. 17 No. 2, pages
	 * 77-87, April 1995.
	 * <p>
	 * W.H. Wolberg, W.N. Street, D.M. Heisey, and O.L. Mangasarian.
	 * Computerized breast cancer diagnosis and prognosis from fine needle
	 * aspirates. Archives of Surgery 1995;130:511-516.
	 * <p>
	 * W.H. Wolberg, W.N. Street, D.M. Heisey, and O.L. Mangasarian.
	 * Computer-derived nuclear features distinguish malignant from benign
	 * breast cytology. Human Pathology, 26:792--796, 1995.
	 * <p>
	 * See also: http://www.cs.wisc.edu/~olvi/uwmp/mpml.html
	 * http://www.cs.wisc.edu/~olvi/uwmp/cancer.html
	 * <p>
	 * Results:
	 * <p>
	 * - predicting field 2, diagnosis: B = benign, M = malignant - sets are
	 * linearly separable using all 30 input features - best predictive accuracy
	 * obtained using one separating plane in the 3-D space of Worst Area, Worst
	 * Smoothness and Mean Texture. Estimated accuracy 97.5% using repeated
	 * 10-fold crossvalidations. Classifier has correctly diagnosed 176
	 * consecutive new patients as of November 1995.
	 * 
	 * <p>
	 * 4. Relevant information
	 * <p>
	 * Features are computed from a digitized image of a fine needle aspirate
	 * (FNA) of a breast mass. They describe characteristics of the cell nuclei
	 * present in the image. A few of the images can be found at
	 * http://www.cs.wisc.edu/~street/images/
	 * <p>
	 * Separating plane described above was obtained using Multisurface
	 * Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree Construction Via
	 * Linear Programming." Proceedings of the 4th Midwest Artificial
	 * Intelligence and Cognitive Science Society, pp. 97-101, 1992], a
	 * classification method which uses linear programming to construct a
	 * decision tree. Relevant features were selected using an exhaustive search
	 * in the space of 1-4 features and 1-3 separating planes.
	 * <p>
	 * The actual linear program used to obtain the separating plane in the
	 * 3-dimensional space is that described in: [K. P. Bennett and O. L.
	 * Mangasarian: "Robust Linear Programming Discrimination of Two Linearly
	 * Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
	 * 
	 * <p>
	 * 5. Number of instances: 569
	 * <p>
	 * 6. Number of attributes: 30 (real-valued input
	 * features)
	 * 
	 * <p>
	 * 7. Attribute information
	 * 
	 * <ul>
	 * <li>1) Diagnosis (0 = malignant, 1 = benign))
	 * </ul>
	 * 
	 * <p>
	 * Ten real-valued features are computed for each cell nucleus:
	 * <ul>
	 * <li>a) radius (mean of distances from center to points on the perimeter) 
	 * <li>b) texture (standard deviation of gray-scale values) 
	 * <li>c) perimeter 
	 * <li>d) area 
	 * <li>e) smoothness (local variation in radius lengths) 
	 * <li>f) compactness (perimeter^2 / area - 1.0) 
	 * <li>g) concavity (severity of concave portions of the contour) 
	 * <li>h) concave points (number of concave portions of the contour) 
	 * <li>i) symmetry 
	 * <li>j) fractal dimension ("coastline approximation" - 1)
	 * </ul>
	 * 
	 * Several of the papers listed above contain detailed descriptions of how
	 * these features are computed.
	 * 
	 * <p>
	 * The mean, standard error, and "worst" or largest (mean of the three
	 * largest values) of these features were computed for each image, resulting
	 * in 30 features.
	 * 
	 * <p>
	 * All feature values are recoded with four significant digits.
	 * 
	 * <p>
	 * 8. Missing attribute values: none
	 * 
	 * <p>
	 * 9. Class distribution: 357 benign, 212 malignant
	 * 
	 * @return the breast cancer dataset
	 */
	public static DataSet loadBreastCancer() {
		return BreastCancerLoader.load();
	}
	
	/**
	 * A simple toy dataset of two crescent-shaped features intertwining.
	 * Good for benchmarking.
	 * @return the toy moons dataset
	 */
	public static DataSet loadToyMoons() {
		return ToyMoonsLoader.load();
	}
}
