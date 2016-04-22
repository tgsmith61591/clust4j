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

import com.clust4j.utils.VecUtils;

public enum Similarity implements SimilarityMetric {
	COSINE {
		@Override public double getDistance(final double[] a, final double[] b) {
			return -getSimilarity(a, b);
		}
		
		@Override public double getSimilarity(final double[] a, final double[] b) {
			return VecUtils.cosSim(a, b);
		}
		
		@Override public String getName() {
			return "Cosine Similarity";
		}

		@Override
		public double getPartialDistance(double[] a, double[] b) {
			return getDistance(a, b);
		}

		@Override
		public double partialDistanceToDistance(double d) {
			return d;
		}

		@Override
		public double distanceToPartialDistance(double d) {
			return d;
		}

		@Override
		public double getPartialSimilarity(double[] a, double[] b) {
			return getSimilarity(a, b);
		}

		@Override
		public double partialSimilarityToSimilarity(double d) {
			return d;
		}

		@Override
		public double similarityToPartialSimilarity(double d) {
			return d;
		}
	},
	
	;
	
	@Override
	public String toString() {
		return getName();
	}
}
