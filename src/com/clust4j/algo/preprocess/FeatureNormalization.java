package com.clust4j.algo.preprocess;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState.FeatureNormalizationConf;
import com.clust4j.except.NonUniformMatrixException;

/**
 * A set of column normalizing enums used 
 * for preprocessing or scaling data.
 * @author Taylor G Smith
 */
public enum FeatureNormalization implements PreProcessor {
	MEAN_CENTER {
		@Override
		public PreProcessor copy() { return this; }

		@Override
		public AbstractRealMatrix operate(AbstractRealMatrix data) {
			return new Array2DRowRealMatrix(operate(data.getData()), false);
		}
		
		@Override
		public double[][] operate(double[][] data) {
			/*
			 * This method gets called tons of times, and needs
			 * to be highly optimized. Thus, we won't use build in
			 * methods that will push it into the O(3M * N) range...
			 * Instead, we'll try to do it in 2M * N.
			 */
			
			final int m = data.length, n = data[0].length;
			final double[][] copy = new double[m][];
			
			for(int col = 0; col < n; col++) {
				double sum = 0.0;
				
				for(int row = 0; row < m; row++) {
					if(data[row].length != n)
						throw new NonUniformMatrixException(n, data[row].length);
					
					// If first iteration, init new double arr
					if(0 == col)
						copy[row] = new double[n];
					
					sum += data[row][col];
				}
				
				double mn  = sum / (double)m;
				
				// Set in place.
				for(int row = 0; row < m; row++)
					copy[row][col] = (data[row][col] - mn);
			}
			
			return copy;
		}
		
	},
	
	/**
	 * Fits the data into the range [0,1] by default. If {@link FeatureNormalizationConf#MIN_MAX_SCALER_RANGE_MIN} or
	 * {@link FeatureNormalizationConf#MIN_MAX_SCALER_RANGE_MAX} is set differently, will fit into range [MIN, MAX]
	 */
	MIN_MAX_SCALE {
		int RANGE_MIN = FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MIN;
		int RANGE_MAX = FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MAX;
		
		@Override
		public PreProcessor copy() { return this; }
		
		@Override
		public AbstractRealMatrix operate(AbstractRealMatrix data) {
			return new Array2DRowRealMatrix(operate(data.getData()), false);
		}
		
		/**
		 * @throws IllegalStateException if {@link FeatureNormalizationConf#MIN_MAX_SCALER_RANGE_MIN} is greater
	     * than or equal to {@link FeatureNormalizationConf#MIN_MAX_SCALER_RANGE_MAX}
		 */
		@Override
		public double[][] operate(double[][] data) {
			/*
			 * This method gets called tons of times, and needs
			 * to be highly optimized. Thus, we won't use build in
			 * methods that will push it into the O(5M * N) range...
			 * Instead, we'll try to do it in 2M * N.
			 */
			
			// Ensure min/max are higher than each other...
			if(RANGE_MIN >= RANGE_MAX)
				throw new IllegalStateException("RANGE_MIN ("+RANGE_MIN+
						") must be lower than RANGE_MAX ("+RANGE_MAX+")");
			
			final int m = data.length, n = data[0].length;
			final double[][] copy = new double[m][];
			
			for(int col = 0; col < n; col++) {
				double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY, rng;
				
				for(int row = 0; row < m; row++) {
					if(data[row].length != n)
						throw new NonUniformMatrixException(n, data[row].length);
					
					// If first iteration, init new double arr
					if(0 == col)
						copy[row] = new double[n];
					
					min = FastMath.min(min, data[row][col]);
					max = FastMath.max(max, data[row][col]);
				}
				
				rng = max - min;
				
				// Set in place.
				for(int row = 0; row < m; row++)
					copy[row][col] = ((data[row][col] - min) / rng) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN;
			}
			
			return copy;
		}
	},
	
	/**
	 * Scales the feature to mean 0, unit variance
	 */
	STANDARD_SCALE {
		@Override
		public PreProcessor copy() { return this; }

		@Override
		public AbstractRealMatrix operate(AbstractRealMatrix data) {
			return new Array2DRowRealMatrix(operate(data.getData()), false);
		}

		@Override
		public double[][] operate(double[][] data) {
			/*
			 * This method gets called tons of times, and needs
			 * to be highly optimized. Thus, we won't use build in
			 * methods that will push it into the O(5M * N) range...
			 * Instead, we'll try to do it in 2M * N.
			 */
			
			/*
			MatUtils.checkDims(data);
			final double[][] copy = MatUtils.copy(data);
			final int m = data.length, n = data[0].length;
			
			for(int col = 0; col < n; col++) {
				final double[] v = MatUtils.getColumn(copy, col);
				final double mean = VecUtils.nanMean(v);
				final double sd = VecUtils.nanStdDev(v, mean);
				
				for(int row = 0; row < m; row++) {
					final double new_val = (v[row] - mean) / sd;
					copy[row][col] = new_val;
				}
			}
			*/
			
			final int m = data.length, n = data[0].length;
			final double[][] copy = new double[m][];
			
			for(int col = 0; col < n; col++) {
				double var, std, mn;
				double sumSq = 0.0;
				double sum   = 0.0;
				
				for(int row = 0; row < m; row++) {
					if(data[row].length != n)
						throw new NonUniformMatrixException(n, data[row].length);
					
					// If first iteration, init new double arr
					if(0 == col)
						copy[row] = new double[n];
					
					sumSq += data[row][col] * data[row][col];
					sum += data[row][col];
				}
				
				/*
				 * A naive algorithm to calculate the estimated variance (1M):
				 * 
				 * Let n = 0, Sum = 0, SumSq = 0 
				 * For each datum x: 
				 *   n = n + 1 
				 *   Sum = Sum + x 
				 *   SumSq = SumSq + x * x 
				 * Var = (SumSq − (Sum × Sum) / n) / (n − 1)
				 * 
				 * @see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
				 */
				var = (sumSq - (sum*sum)/(double)m ) / ((double)m - 1.0);
				std = m < 2 ? Double.NaN : FastMath.sqrt(var);
				mn  = sum / (double)m;
				
				// Set in place.
				for(int row = 0; row < m; row++)
					copy[row][col] = (data[row][col] - mn) / std;
			}
			
			return copy;
		}
		
	},
	
}
