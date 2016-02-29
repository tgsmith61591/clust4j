package com.clust4j.algo;

interface ConvergeablePlanner extends java.io.Serializable {
	/**
	 * The maximum number of iterations the algorithm
	 * is permitted before aborting without converging
	 * @return max iterations before convergence
	 */
	public int getMaxIter();
	
	/**
	 * This minimum change between iterations that will
	 * denote an iteration as having converged
	 * @return the min change for convergence
	 */
	public double getConvergenceTolerance();
}
