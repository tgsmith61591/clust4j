package com.clust4j.algo.preprocess;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.GlobalState;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

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
			MatUtils.checkDims(data);
			final double[][] copy = MatUtils.copyMatrix(data);
			final int n = data[0].length;
			
			for(int col = 0; col < n; col++) {
				final double[] v = MatUtils.getColumn(copy, col);
				MatUtils.setColumnInPlace(copy, col, VecUtils.scalarSubtract(v, VecUtils.mean(v)));
			}
			
			return copy;
		}
		
	},
	
	/**
	 * Fits the data into the range [0,1] by default. If {@link MIN_MAX_SCALE#RANGE_MIN} or
	 * {@link MIN_MAX_SCALE#RANGE_MAX} is set differently, will fit into range [MIN, MAX]
	 */
	MIN_MAX_SCALE {
		int RANGE_MIN = GlobalState.FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MIN;
		int RANGE_MAX = GlobalState.FeatureNormalizationConf.MIN_MAX_SCALER_RANGE_MAX;
		
		@Override
		public PreProcessor copy() { return this; }
		
		@Override
		public AbstractRealMatrix operate(AbstractRealMatrix data) {
			return new Array2DRowRealMatrix(operate(data.getData()), false);
		}
		
		@Override
		public double[][] operate(double[][] data) {
			MatUtils.checkDims(data);
			
			// Ensure min/max are higher than each other...
			if(RANGE_MIN >= RANGE_MAX)
				throw new IllegalStateException("RANGE_MIN ("+RANGE_MIN+
						") must be lower than RANGE_MAX ("+RANGE_MAX+")");
			
			final double[][] copy = MatUtils.copyMatrix(data);
			final int m = data.length, n = data[0].length;
			
			for(int col = 0; col < n; col++) {
				final double[] v = MatUtils.getColumn(copy, col);
				final double max = VecUtils.max(v);
				final double min = VecUtils.min(v);
				final double rng = (max - min);
				
				for(int row = 0; row < m; row++)
					copy[row][col] = ((copy[row][col] - min) / rng) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN;
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
			MatUtils.checkDims(data);
			final double[][] copy = MatUtils.copyMatrix(data);
			final int m = data.length, n = data[0].length;
			
			for(int col = 0; col < n; col++) {
				final double[] v = MatUtils.getColumn(copy, col);
				final double mean = VecUtils.mean(v);
				final double sd = VecUtils.stdDev(v, mean);
				
				for(int row = 0; row < m; row++) {
					final double new_val = (v[row] - mean) / sd;
					copy[row][col] = new_val;
				}
			}
			
			return copy;
		}
		
	},
	
}
