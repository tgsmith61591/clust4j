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
package com.clust4j.algo;

import java.util.Collection;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.except.ModelNotFitException;
import com.clust4j.algo.NearestCentroid.NearestCentroidPlanner;

/**
 * A standalone mixin class to handle predictions from {@link CentroidLearner}
 * classes that are also a {@link BaseClassifier} and a subclass of {@link AbstractClusterer}.
 * @author Taylor G Smith
 */
abstract class CentroidUtils {
	
	/**
	 * Returns a matrix with the centroids.
	 * @param copy - whether or not to keep the reference or copy
	 * @return Array2DRowRealMatrix
	 */
	protected static Array2DRowRealMatrix centroidsToMatrix(final Collection<double[]> centroids, boolean copy) {
		double[][] c = new double[centroids.size()][];
		
		int i = 0;
		for(double[] row: centroids)
			c[i++] = row;
		
		return new Array2DRowRealMatrix(c, copy);
	}
	
	/**
	 * Predict on an already-fit estimator
	 * @param model
	 * @param X
	 * @throws ModelNotFitException if the model isn't fit
	 */
	protected static <E extends AbstractClusterer & CentroidLearner & BaseClassifier>
			int[] predict(E model, AbstractRealMatrix newData) throws ModelNotFitException {
		
		/*
		 * First get the ground truth from the estimator...
		 */
		final int[] labels = model.getLabels(); // throws exception
		
		/*
		 * Now fit the NearestCentroids model, and predict
		 */
		return new NearestCentroidPlanner()
				.setMetric(model.dist_metric) // if it fails, falls back to default Euclidean...
				.setVerbose(false) // just to be sure in case default ever changes...
				.buildNewModelInstance(model.getData(), labels)
			.fit()
			.predict(newData);
	}
}
