package com.clust4j.algo.prep;

import java.util.Random;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.log.Log;
import com.clust4j.log.Loggable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.NaNException;
import com.clust4j.utils.Named;

public abstract class MatrixImputation implements Loggable, Named, PreProcessor {
	final public static boolean DEF_VERBOSE = AbstractClusterer.DEF_VERBOSE;
	private boolean verbose = DEF_VERBOSE;
	private Random seed = new Random();
	
	
	
	public MatrixImputation() { }
	
	public MatrixImputation(final ImputationPlanner planner) {
		this.verbose = planner.getVerbose();
		this.seed = planner.getSeed();
	}
	
	
	
	abstract static public class ImputationPlanner {
		public ImputationPlanner(){}
		abstract public Random getSeed();
		abstract public boolean getVerbose();
		abstract public ImputationPlanner setSeed(Random rand);
		abstract public ImputationPlanner setVerbose(boolean b);
	}
	
	
	
	
	
	/**
	 * Call this prior to every process call!
	 * @param data
	 */
	protected final void checkMat(final double[][] data) {
		MatUtils.checkDims(data);
		final int m = data.length, n = data[0].length;
		
		// Now check column NaN level
		boolean seenNaN = false;
		final double[][] dataCopy = MatUtils.copyMatrix(data);
		for(int col = 0; col < n; col++) {
			Inner:
			for(int row = 0; row < m; row++) {
				boolean nan = Double.isNaN(dataCopy[row][col]);
				if(nan) {
					seenNaN =true;
					if(row == m - 1)
						throw new NaNException("column " + col + " is entirely NaN");
				} else break Inner;
			}
		}
		
		if(!seenNaN) warn("no NaNs in matrix; imputation will not have any effect");
		
		// TODO?
		
		info("initializing matrix imputation method");
	}
	
	public Random getSeed() {
		return seed;
	}
	
	
	/* -- LOGGER METHODS --  */
	@Override public void error(String msg) {
		if(verbose) Log.err(getLoggerTag(), msg);
	}
	
	@Override public void warn(String msg) {
		if(verbose) Log.warn(getLoggerTag(), msg);
	}
	
	@Override public void info(String msg) {
		if(verbose) Log.info(getLoggerTag(), msg);
	}
	
	@Override public void trace(String msg) {
		if(verbose) Log.trace(getLoggerTag(), msg);
	}
	
	@Override public void debug(String msg) {
		if(verbose) Log.debug(getLoggerTag(), msg);
	}
}
