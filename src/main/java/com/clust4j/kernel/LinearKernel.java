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

import com.clust4j.utils.VecUtils;

/**
 * The Linear kernel is the simplest kernel function. 
 * It is given by the inner product <tt>&lt;x,y&gt;</tt> plus an optional constant c. 
 * Kernel algorithms using a linear kernel are often equivalent to 
 * their non-kernel counterparts, i.e. 
 * 
 * <a href="http://crsouza.com/2010/03/kernel-functions-for-machine-learning-applications/2010/01/kernel-principal-component-analysis-in-c/">kernel principal component analysis</a> 
 * with linear kernel is the same as standard PCA.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class LinearKernel extends ConstantKernel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -9140596365379085676L;
	public static final double DEFAULT_LIN_CONSTANT = 0;

	public LinearKernel() { this(DEFAULT_LIN_CONSTANT); }
	public LinearKernel(final double constant) {
		super(constant);
	}
	
	@Override
	public double getSimilarity(final double[] a, final double[] b) {
		return VecUtils.innerProduct(a, b) + getConstant();
	}
	
	@Override
	public String getName() {
		return "LinearKernel";
	}
}
