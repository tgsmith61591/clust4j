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
package com.clust4j.sample;

import java.util.Random;
import java.util.TreeSet;

import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

/**
 * <a href="https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29">Bootstrapping</a> 
 * can refer to any test or metric that relies on random sampling with replacement. 
 * Bootstrapping allows assigning measures of accuracy (defined in terms of bias, 
 * variance, confidence intervals, prediction error or some other such measure) to sample estimates. 
 * In the context of clust4j, the Bootstrapper only provides an interface for random
 * (or seeded) sampling with replacement, as it implements {@link Sampler}.
 * 
 * <p>The internal use of the Bootstrapper class is for use with {@link MatrixImputation}.
 * @author Taylor G Smith
 *
 */
public enum Bootstrapper implements Sampler, java.io.Serializable {
	
	/**
	 * Performs uniformly random (or seeded) sampling across a matrix.
	 */
	BASIC {
		@Override public String getName() {
			return "Uniform Bootstrapper";
		}
		
		@Override public double[][] sample(final double[][] data, final int n) {
			return sample(data, n, new Random());
		}
		
		@Override public double[][] sample(final double[][] data, final int n, final Random seed) {
			MatUtils.checkDims(data);
			
			final int m = data.length;
			final double[][] out = new double[n][];
			for(int i = 0; i < n; i++)
				out[i] = VecUtils.copy(data[seed.nextInt(m)]);
			
			return out;
		}
		
		@Override
		public String toString() {
			return getName();
		}
	},
	
	/**
	 * Performs uniform bootstrapping across a matrix while 
	 * adding a very small amount of random gaussian noise to resampled
	 * records. That is, the first time a record is sampled, no gaussian
	 * noise will be applied, but subsequent samplings of the same record
	 * will have added noise.
	 */
	SMOOTH {
		@Override public String getName() {
			return "Smooth Bootstrapper";
		}
		
		@Override public double[][] sample(final double[][] data, final int n) {
			return sample(data, n, new Random());
		}
		
		@Override public double[][] sample(final double[][] data, final int n, final Random seed) {
			MatUtils.checkDims(data);
			
			final int m = data.length, k = data[0].length;
			final TreeSet<Integer> seen = new TreeSet<>();
			final double[][] out = new double[n][];
			for(int i = 0; i < n; i++) {
				int next = seed.nextInt(m);
				
				if(seen.contains(next)) // Already sampled this... add the noise
					out[i] = VecUtils.add(VecUtils.copy(data[next]), 
							VecUtils.randomGaussianNoiseVector(k, seed));
				else {
					out[i] = VecUtils.copy(data[next]);
					seen.add(next);
				}
			}
			
			return out;
		}
		
		@Override
		public String toString() {
			return getName();
		}
	},
}
