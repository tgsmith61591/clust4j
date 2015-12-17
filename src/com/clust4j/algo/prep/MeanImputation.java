package com.clust4j.algo.prep;

import java.util.Random;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.MatUtils;

public class MeanImputation extends MatrixImputation {
	
	public MeanImputation() {
		this(new MeanImputationPlanner());
	}
	
	public MeanImputation(MeanImputationPlanner planner) {
		super(planner);
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
	public String getName() {
		return "Mean imputation";
	}
	
	@Override
	public double[][] process(final double[][] dat) {
		checkMat(dat);
		
		final double[][] copy = MatUtils.copyMatrix(dat);
		final int m = dat.length, n = dat[0].length;
		info("(" + getName() + ") performing mean imputation on " + m + " x " + n + " dataset");
		
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
			
			info("(" + getName() + ") " + nanCt + " NaN" + (nanCt!=1?"s":"") + " identified in column " + col + " (column mean="+mean+")");
		}
		
		return copy;
	}
}
