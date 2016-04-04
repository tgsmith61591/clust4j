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
