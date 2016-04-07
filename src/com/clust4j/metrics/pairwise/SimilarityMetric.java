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
package com.clust4j.metrics.pairwise;

public interface SimilarityMetric extends GeometricallySeparable, java.io.Serializable { 
	/**
	 * Generally equal to negative {@link #getDistance(double[], double[])}
	 * @param a
	 * @param b
	 * @return
	 */
	public double getSimilarity(final double[] a, final double[] b);
	
	/**
	 * If the metric allows for a faster, non-canonical computation
	 * that will maintain ordinality in similarity computations,
	 * this method will compute it. Otherwise, it should return
	 * {@link #getSimilarity(double[], double[])}. 
	 * 
	 * @param a
	 * @param b
	 * @return the partial similarity
	 */
	public double getPartialSimilarity(final double[] a, final double[] b);
	
	/**
	 * Convert the partial similarity to the full similarity
	 * @param a
	 * @param b
	 * @return the full distance
	 */
	public double partialSimilarityToSimilarity(double d);
	
	/**
	 * Convert the full similarity to the partial similarity
	 * @param a
	 * @param b
	 * @return the partial distance
	 */
	public double similarityToPartialSimilarity(double d);
}
