package com.clust4j.sample;

import java.util.Random;

import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public enum Bootstrapper implements Sampler {
	
	UNIFORM {
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
			final double[][] out = new double[n][];
			for(int i = 0; i < n; i++)
				out[i] = VecUtils.add(VecUtils.copy(data[seed.nextInt(m)]), 
					VecUtils.randomGaussianNoiseVector(k, seed));
			
			return out;
		}
		
		@Override
		public String toString() {
			return getName();
		}
	},
	
	//TODO BAYESIAN
}
