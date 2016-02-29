package com.clust4j.algo;

/**
 * An interface to be implemented by {@link AbstractAutonomousClusterer}<tt>s</tt> that converge
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 */
public interface Convergeable extends ConvergeablePlanner {
	/**
	 * Returns whether the algorithm has converged yet.
	 * If the algorithm has yet to be fit, it will return false.
	 * @return the state of algorithmic convergence
	 */
	public boolean didConverge();
	
	/**
	 * 
	 * @return
	 */
	public int itersElapsed();
}
