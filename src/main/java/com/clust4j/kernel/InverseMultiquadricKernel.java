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
 * The Inverse {@link MultiQuadricKernel}. As with the {@link GaussianKernel}, 
 * it results in a kernel matrix with full rank 
 * <a href="http://www.springerlink.com/content/w62233k766460945/">(Micchelli, 1986)</a> and thus 
 * forms an infinite dimension feature space.
 * 
 * <p>If <tt>||<i>x</i> - <i>y</i>|| < SIGMA</tt>, the similarity is 0.
 * 
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class InverseMultiquadricKernel extends MultiquadricKernel {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -7294670048769421427L;
	public InverseMultiquadricKernel() {
		super();
	}
	
	public InverseMultiquadricKernel(final double constant) {
		super(constant);
	}
	
	@Override
	public String getName() {
		return "InverseMultiquadricKernel";
	}
	
	@Override
	public double getSimilarity(final double[] a, final double[] b) {
		return 1.0 / super.getSimilarity(a, b);
	}
}
