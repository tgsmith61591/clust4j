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

/**
 * The exponential kernel is closely related to the {@link GaussianKernel}, 
 * with only the square of the norm left out. It is also a radial basis 
 * function kernel.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class ExponentialKernel extends LaplacianKernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4364593461130945118L;
	public static final double DEFAULT_EXPONENTIAL	= 1;
	public static final double DEFAULT_SIGMA_EXP	= 2;
	public static final double DEFAULT_SIGMA_SCALAR	= 2;
	
	public ExponentialKernel() {
		this(DEFAULT_SIGMA);
	}
	
	public ExponentialKernel(final double sigma) {
		this(sigma, DEFAULT_EXPONENTIAL);
	}
	
	/**
	 * For use with GaussianKernal
	 * @param SIGMA
	 * @param EXPONENTIAL
	 */
	protected ExponentialKernel(final double SIGMA, final double EXPONENTIAL) {
		super(SIGMA, EXPONENTIAL, DEFAULT_SIGMA_EXP, DEFAULT_SIGMA_SCALAR);
	}
	
	@Override
	public String getName() {
		return "ExponentialKernel";
	}
}
