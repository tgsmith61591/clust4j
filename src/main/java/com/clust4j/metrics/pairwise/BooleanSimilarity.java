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
package com.clust4j.metrics.pairwise;

import com.clust4j.utils.QuadTup;
import com.clust4j.utils.VecUtils;

/**
 * A helper class for boolean dissimilarity metrics like {@link Distance#RUSSELL_RAO},
 * {@link Distance#DICE}, etc. Any non-zero elements are treated as true, and otherwise false.
 * Position one is count of TT, two is TF, three is FT and four is FF.
 * @author Taylor G Smith
 */
class BooleanSimilarity extends QuadTup<Double, Double, Double, Double> {
	private static final long serialVersionUID = 6735795579759248156L;

	private BooleanSimilarity(Double one, Double two, Double three, Double four) {
		super(one, two, three, four);
	}

	static BooleanSimilarity build(final double[] a, final double[] b) {
		VecUtils.checkDims(a, b);
		double ctt = 0.0, ctf = 0.0, cft = 0.0, cff = 0.0;
		
		for(int i = 0; i < a.length; i++) {
			if(a[i] != 0 && b[i] != 0)
				ctt += 1.0;
			else if(a[i] != 0)
				ctf += 1.0;
			else if(b[i] != 0)
				cft += 1.0;
			else 
				cff += 1.0;
		}
		
		return new BooleanSimilarity(ctt, ctf, cft, cff);
	}
}
