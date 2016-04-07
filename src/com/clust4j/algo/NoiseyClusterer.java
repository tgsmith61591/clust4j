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

/**
 * Any cluster that does not force a prediction for every
 * single point is considered a "noisey clusterer." This interface
 * provides the method {@link #getNumberOfNoisePoints()}, which
 * returns the number of points that were not classified as
 * belonging to any clusters.
 * 
 * @author Taylor G Smith
 * @see {@link DBSCAN}
 * @see {@link MeanShift}
 */
public interface NoiseyClusterer extends java.io.Serializable {
	final public static int NOISE_CLASS = -1;
	
	/**
	 * the number of points that were not classified as
	 * belonging to any clusters.
	 * @return how many points are considered noise
	 */
	public int getNumberOfNoisePoints();
}
