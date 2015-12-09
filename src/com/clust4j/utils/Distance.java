package com.clust4j.utils;

import org.apache.commons.math3.util.FastMath;

public enum Distance implements GeometricallySeparable, java.io.Serializable {
	MANHATTAN {
		@Override 
		public double getDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			double sum = 0;
			for(int i = 0; i < a.length; i++) {
				// Don't use math.abs -- too expensive
				double diff = a[i] - b[i];
				sum += FastMath.abs(diff);
			}
			
			return sum;
		}
		
		@Override
		public String getName() {
			return "Manhattan";
		}
	},
	
	EUCLIDEAN {
		@Override 
		public double getDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			double sum = 0;
			for(int i = 0; i < a.length; i++) {
				// Don't use math.pow -- too expensive
				double diff = a[i]-b[i];
				sum += diff * diff;
			}
			
			return FastMath.sqrt(sum);
		}
		
		@Override
		public String getName() {
			return "Euclidean";
		}
	}
}
