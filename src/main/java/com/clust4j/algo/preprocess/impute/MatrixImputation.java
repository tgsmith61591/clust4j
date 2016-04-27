/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/

package com.clust4j.algo.preprocess.impute;

import java.util.Random;

import com.clust4j.NamedEntity;
import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.preprocess.PreProcessor;
import com.clust4j.except.NaNException;
import com.clust4j.log.Log;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Loggable;
import com.clust4j.utils.MatUtils;

/**
 * {@link AbstractClusterer} algorithms are not capable of robustly handling
 * missing values (or {@link Double#NaN} in clust4j). If an algorithm is invoked
 * on missing data, a {@link NaNException} will be thrown. To rectify these missing
 * values, this class and its children are designed to impute the missing values
 * using different statistical metrics.
 * 
 * @author Taylor G Smith
 */
public abstract class MatrixImputation extends PreProcessor implements Loggable, NamedEntity {
	private static final long serialVersionUID = 8816387041123292806L;
	
	final public static boolean DEF_VERBOSE = AbstractClusterer.DEF_VERBOSE;
	protected boolean verbose = DEF_VERBOSE;
	private Random seed = new Random();
	private boolean hasWarnings = false;
	
	
	
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
		final double[][] dataCopy = MatUtils.copy(data);
		for(int col = 0; col < n; col++) {
			Inner:
			for(int row = 0; row < m; row++) {
				boolean nan = Double.isNaN(dataCopy[row][col]);
				if(nan) {
					seenNaN =true;
					if(row == m - 1)
						error(new NaNException("column " + 
							col + " is entirely NaN"));
					
				} else break Inner;
			}
		}
		
		if(!seenNaN) warn("no NaNs in matrix; imputation will not have any effect");
		info("initializing matrix imputation method");
	}
	
	public Random getSeed() {
		return seed;
	}
	
	
	/* -- LOGGER METHODS --  */
	@Override public void error(String msg) {
		if(verbose) Log.err(getLoggerTag(), msg);
	}
	
	@Override public void error(RuntimeException thrown) {
		error(thrown.getMessage());
		throw thrown;
	}
	
	@Override public void warn(String msg) {
		hasWarnings = true;
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
	
	@Override
	public boolean hasWarnings() {
		return hasWarnings;
	}
	
	/**
	 * Write the time the algorithm took to complete
	 * @param timer
	 */
	@Override public void sayBye(final LogTimer timer) {
		info("imputation task completed in " + timer.toString());
	}
}
