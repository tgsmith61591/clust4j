package com.clust4j.algo.prep;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.Log.Tag.Algo;

public class MeanImputation extends MatrixImputation {
	
	
	public MeanImputation(AbstractRealMatrix data) {
		super(data);
	}
	
	public MeanImputation(final double[][] data) {
		super(data);
	}
	
	public MeanImputation(final AbstractRealMatrix data, final boolean copy) {
		super(data, copy);
	}
	
	public MeanImputation(AbstractRealMatrix data, MeanImputationPlanner planner) {
		super(data, planner);
	}
	
	public MeanImputation(final double[][] data, MeanImputationPlanner planner) {
		super(data, planner);
	}
	
	public MeanImputation(final AbstractRealMatrix data, final boolean copy, MeanImputationPlanner planner) {
		super(data, copy, planner);
	}
	

	
	
	public static class MeanImputationPlanner extends ImputationPlanner {
		private boolean verbose = DEF_VERBOSE;
		private Random seed = new Random();
		
		public MeanImputationPlanner() {}

		@Override
		public Random getSeed() {
			return seed;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		@Override
		public MeanImputationPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}

		@Override
		public MeanImputationPlanner setVerbose(boolean b) {
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
		info("performing mean imputation on " + m + " x " + n + " dataset");
		
		// Operates in 2M * N
		for(int col = 0; col < n; col++) {
			int count = 0;
			double sum = 0;
			for(int row = 0; row < m; row++) {
				if(!Double.isNaN(copy[row][col])) {
					sum += copy[row][col];
					count++;
				}
			}

			int nanCt = m - count;
			double mean = sum / (double)count;
			for(int row = 0; row < m; row++) {
				if(Double.isNaN(copy[row][col])) {
					copy[row][col] = mean;
				}
			}
			
			info(nanCt + " NaN" + (nanCt!=1?"s":"") + " identified in column " + col + " (column mean="+mean+")");
		}
		
		return copy;
	}
}
