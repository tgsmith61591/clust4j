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
package com.clust4j.kernel;

import org.apache.commons.math3.util.FastMath;

/**
 * The Log kernel seems to be particularly interesting for 
 * images, but is only conditionally positive definite
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class LogKernel extends PowerKernel {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1059869495129543995L;
	public LogKernel() {
		super();
	}
	
	public LogKernel(final double degree) {
		super(degree);
	}
	
	@Override
	public String getName() {
		return "LogKernel";
	}
	
	@Override
	public double getSimilarity(final double[] a, final double[] b) {
		final double sup = -(super.getSimilarity(a, b)); // super returns negative, so reverse it
		final double answer = -FastMath.log(sup + 1);
		return Double.isNaN(answer) ? Double.NEGATIVE_INFINITY : answer;
	}
}
