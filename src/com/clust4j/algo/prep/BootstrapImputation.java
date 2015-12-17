package com.clust4j.algo.prep;

import java.util.Random;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.sample.Bootstrapper;
import com.clust4j.utils.IllegalClusterStateException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.NaNException;
import com.clust4j.utils.VecUtils;

public class BootstrapImputation extends MatrixImputation {
	final static public double DEF_RATIO = 0.67;
	final static public Bootstrapper DEF_BOOTSTRAPPER = Bootstrapper.UNIFORM;
	final static public CentralTendencyMethod DEF_CENT_METHOD = CentralTendencyMethod.MEAN;
	private CentralTendencyMethod ctm = DEF_CENT_METHOD;
	private Bootstrapper strap = DEF_BOOTSTRAPPER;
	private double ratio = DEF_RATIO;
	
	
	public static enum CentralTendencyMethod {
		MEAN, 
		MEDIAN
	}
	
	
	
	public BootstrapImputation() {
		this(new BootstrapImputationPlanner());
	}
	
	public BootstrapImputation(BootstrapImputationPlanner planner) {
		super(planner);
		initFromPlanner(planner);
	}
	
	
	private void initFromPlanner(BootstrapImputationPlanner planner) {
		this.ctm = planner.method;
		this.strap = planner.strap;
		this.ratio = planner.ratio;

		if(ratio < 0 || ratio > 1)
			throw new IllegalArgumentException("ratio must be between 0 and 1");
		if(null == strap)
			throw new IllegalClusterStateException("null bootstrapper");
		
		info("central tendency="+ctm);
		info("bootstrapper="+strap);
		info("sampling ratio="+ratio);
	}
	

	
	
	public static class BootstrapImputationPlanner extends ImputationPlanner {
		private boolean verbose = DEF_VERBOSE;
		private CentralTendencyMethod method = DEF_CENT_METHOD;
		private Bootstrapper strap = DEF_BOOTSTRAPPER;
		private Random seed = new Random();
		private double ratio = DEF_RATIO;
		
		public BootstrapImputationPlanner() {}
		
		@Override
		public Random getSeed() {
			return seed;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		public BootstrapImputationPlanner setBootstrapper(final Bootstrapper strap) {
			this.strap = strap;
			return this;
		}
		
		public BootstrapImputationPlanner setMethodOfCentralTendency(final CentralTendencyMethod method) {
			this.method = method;
			return this;
		}
		
		public BootstrapImputationPlanner setRatio(final double ratio) {
			this.ratio = ratio;
			return this;
		}
		
		@Override
		public BootstrapImputationPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}

		@Override
		public BootstrapImputationPlanner setVerbose(boolean b) {
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
		return strap.getName() + " imputation";
	}
	
	@Override
	public double[][] process(final double[][] dat) {
		checkMat(dat);
		
		final boolean mean = ctm.equals(CentralTendencyMethod.MEAN);
		final double[][] complete = MatUtils.completeCases(dat);
		
		String error;
		if(complete.length == 0) {
			error = "(" + getName() + ") no complete records in matrix";
			error(error);
			throw new NaNException(error);
		}
		
		
		final int m = dat.length, n = dat[0].length;
		final int mc = complete.length;
		final int ms = (int)Math.ceil(ratio * mc);
		final double[][] sampled = strap.sample(complete, ms, getSeed());

		
		info("(" + getName() + ") performing bootstrap imputation on " + m + " x " + n + " dataset");
		info("(" + getName() + ") " + mc+" complete records found in matrix, "+ms+" records sampled for imputation");
		final double[][] copy = MatUtils.copyMatrix(dat);
		
		
		for(int col = 0; col < n; col++) {
			double val;
			
			if(mean) {
				double sum = 0;
				for(int row = 0; row < ms; row++)
					sum += sampled[row][col];
				val = sum / (double)ms;
			} else {
				val = VecUtils.median(MatUtils.getColumn(sampled, col));
			}
			
			// Impute
			int nanCt = 0;
			for(int row = 0; row < m; row++) {
				if(Double.isNaN(copy[row][col])) {
					copy[row][col] = val;
					nanCt++;
				}
			}
			
			info("(" + getName() + ") " + nanCt + " NaN" + (nanCt!=1?"s":"") + " identified in column " + col + " (imputation value="+mean+")");
		}
		
		return copy;
	}
}
