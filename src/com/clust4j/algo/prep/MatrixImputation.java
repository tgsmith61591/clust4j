package com.clust4j.algo.prep;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.log.Log;
import com.clust4j.log.Loggable;
import com.clust4j.utils.NaNException;

public abstract class MatrixImputation implements Loggable {
	final public static boolean DEF_VERBOSE = AbstractClusterer.DEF_VERBOSE;
	protected final AbstractRealMatrix data;
	private boolean verbose = DEF_VERBOSE;
	private boolean hasWarnings = false;
	private Random seed = new Random();
	
	
	
	public MatrixImputation(final AbstractRealMatrix data) {
		this(data, true);
	}
	
	public MatrixImputation(final double[][] data) {
		this(new Array2DRowRealMatrix(data), false);
	}
	
	public MatrixImputation(final AbstractRealMatrix data, final boolean copy) {
		this.data = copy ? (AbstractRealMatrix)data.copy() : data;
		checkMat(data);
	}
	
	public MatrixImputation(final AbstractRealMatrix data, final ImputationPlanner planner) {
		this(data, true, planner);
	}
	
	public MatrixImputation(final double[][] data, final ImputationPlanner planner) {
		this(new Array2DRowRealMatrix(data), false, planner);
	}
	
	public MatrixImputation(final AbstractRealMatrix data, final boolean copy, final ImputationPlanner planner) {
		this(data, copy);
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
	
	
	
	
	
	final void checkMat(final AbstractRealMatrix data) {
		int m, n;
		if((m = data.getRowDimension()) < 1)
			throw new IllegalArgumentException("input data of length 0");
		if((n = data.getColumnDimension()) < 1)
			throw new IllegalArgumentException("input data with null column space");
		
		// Now check column NaN level
		boolean seenNaN = false;
		final double[][] dataCopy = data.getData();
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
		
		if(!seenNaN) {
			if(verbose) warn("no NaNs in matrix; imputation will not have any effect");
			else flagWarning();
		}
		
		// TODO?
		
		if(verbose) info("initializing matrix imputation method");
	}
	
	public Random getSeed() {
		return seed;
	}
	
	public boolean hasWarnings() {
		return hasWarnings;
	}
	
	private void flagWarning() {
		hasWarnings = true;
	}
	
	
	/* -- LOGGER METHODS --  */
	@Override public void error(String msg) {
		if(verbose) Log.err(getLoggerTag(), msg);
	}
	
	@Override public void warn(String msg) {
		flagWarning();
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
	
	abstract public double[][] impute();
}
