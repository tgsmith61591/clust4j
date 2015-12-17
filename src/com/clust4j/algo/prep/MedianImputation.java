package com.clust4j.algo.prep;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class MedianImputation extends MatrixImputation {
	
	
	public MedianImputation(AbstractRealMatrix data) {
		super(data);
	}
	
	public MedianImputation(final double[][] data) {
		super(data);
	}
	
	public MedianImputation(final AbstractRealMatrix data, final boolean copy) {
		super(data, copy);
	}
	
	public MedianImputation(AbstractRealMatrix data, MedianImputationPlanner planner) {
		super(data, planner);
	}
	
	public MedianImputation(final double[][] data, MedianImputationPlanner planner) {
		super(data, planner);
	}
	
	public MedianImputation(final AbstractRealMatrix data, final boolean copy, MedianImputationPlanner planner) {
		super(data, copy, planner);
	}
	

	
	
	public static class MedianImputationPlanner extends ImputationPlanner {
		private boolean verbose = DEF_VERBOSE;
		private Random seed = new Random();
		
		public MedianImputationPlanner() {}

		@Override
		public Random getSeed() {
			return seed;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		@Override
		public MedianImputationPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}

		@Override
		public MedianImputationPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}
		
	}




	@Override
	public Algo getLoggerTag() {
		return Algo.IMPUTE;
	}

	@Override
	public double[][] impute() {
		final double[][] copy = data.getData();
		final int m = data.getRowDimension(), n = data.getColumnDimension();
		info("performing median imputation on " + m + " x " + n + " dataset");
		
		// Operates in 2M * N
		for(int col = 0; col < n; col++) {
			final double median = VecUtils.nanMedian(MatUtils.getColumn(copy, col));

			int count = 0;
			for(int row = 0; row < m; row++) {
				if(Double.isNaN(copy[row][col])) {
					copy[row][col] = median;
					count++;
				}
			}
			
			info(count + " NaN" + (count!=1?"s":"") + " identified in column " + col + " (column median="+median+")");
		}
		
		return copy;
	}
}
