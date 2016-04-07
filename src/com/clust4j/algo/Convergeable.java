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
package com.clust4j.algo;

/**
 * An interface to be implemented by {@link AbstractAutonomousClusterer}<tt>s</tt> that converge
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 */
public interface Convergeable extends ConvergeablePlanner {
	public static final double DEF_TOL = 0.0;
	
	/**
	 * Returns whether the algorithm has converged yet.
	 * If the algorithm has yet to be fit, it will return false.
	 * @return the state of algorithmic convergence
	 */
	public boolean didConverge();
	
	/**
	 * Get the count of iterations performed by the <tt>fit()</tt> method
	 * @return how many iterations were performed
	 */
	public int itersElapsed();
}
