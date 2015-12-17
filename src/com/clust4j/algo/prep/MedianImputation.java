package com.clust4j.algo.prep;

import java.util.Random;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class MedianImputation extends MatrixImputation {
	
	public MedianImputation() {
		this(new MedianImputationPlanner());
	}
	
	public MedianImputation(MedianImputationPlanner planner) {
		super(planner);
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
	public String getName() {
		return "Median imputation";
	}
	
	@Override
	public double[][] process(final double[][] dat) {
		checkMat(dat);
		
		final double[][] copy = MatUtils.copyMatrix(dat);
		final int m = dat.length, n = dat[0].length;
		info("(" + getName() + ") performing median imputation on " + m + " x " + n + " dataset");
		
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
			
			info("(" + getName() + ") " + count + " NaN" + (count!=1?"s":"") + " identified in column " + col + " (column median="+median+")");
		}
		
		return copy;
	}
}
