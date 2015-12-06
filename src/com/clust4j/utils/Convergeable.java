package com.clust4j.utils;

/**
 * An interface to be implemented by {@link AbstractAutonomousClusterer}<tt>s</tt> that converge
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 */
public interface Convergeable {
	/**
	 * Returns whether the algorithm has converged yet.
	 * If the algorithm has yet to be fit, it will return false.
	 * @return the state of algorithmic convergence
	 */
	public boolean didConverge();
	
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
	public double getMinChange();
	
	/**
	 * 
	 * @return
	 */
	public int itersElapsed();
}
