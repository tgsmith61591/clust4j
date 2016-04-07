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

import com.clust4j.NamedEntity;

public interface GeometricallySeparable extends java.io.Serializable, NamedEntity {
	
	/**
	 * Generally equal to negative {@link #getSimilarity(double[], double[])}
	 * @param a
	 * @param b
	 * @return
	 */
	public double getDistance(final double[] a, final double[] b);
	
	/**
	 * If the metric allows for a faster, non-canonical computation
	 * that will maintain ordinality in distance computations,
	 * this method will compute it. Otherwise, it should return
	 * {@link #getDistance(double[], double[])}. 
	 * 
	 * E.g., for {@link Distance#EUCLIDEAN},
	 * the partial distance method will not compute the sqrt as the final
	 * stage for the sake of efficiency.
	 * @param a
	 * @param b
	 * @return the partial distance
	 */
	public double getPartialDistance(final double[] a, final double[] b);
	
	/**
	 * Convert the partial distance to the full distance
	 * @param a
	 * @param b
	 * @return the full distance
	 */
	public double partialDistanceToDistance(double d);
	
	/**
	 * Convert the full distance to the partial distance
	 * @param a
	 * @param b
	 * @return the partial distance
	 */
	public double distanceToPartialDistance(double d);
}
