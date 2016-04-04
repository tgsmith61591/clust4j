package com.clust4j.metrics.pairwise;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.VecUtils;

public enum Distance implements DistanceMetric, java.io.Serializable {
	HAMMING {
		
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a, b);
			
			final int n = a.length;
			double ct = 0;
			for(int i = 0; i < n; i++)
				if(a[i] != b[i])
					ct++;
			
			return ct / n;
		}
		
		@Override
		final public double getP() {
			return DEFAULT_P;
		}
		
		@Override
		public String getName() {
			return "Hamming";
		}
	},
	
	MANHATTAN {
		
		@Override 
		public double getPartialDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			
			double sum = 0;
			for(int i = 0; i < a.length; i++) {
				double diff = a[i] - b[i];
				sum += FastMath.abs(diff);
			}
			
			return sum;
		}
		
		@Override
		final public double getP() {
			return 1.0;
		}
		
		@Override
		public String getName() {
			return "Manhattan";
		}
	},
	
	
	
	EUCLIDEAN {
		
		@Override
		public double distanceToPartialDistance(final double d) {
			return d * d;
		}
		
		@Override
		final public double getP() {
			return 2.0;
		}
		
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			
			double sum = 0;
			for(int i = 0; i < a.length; i++) {
				// Don't use math.pow -- too expensive
				double diff = a[i]-b[i];
				sum += diff * diff;
			}
			
			return sum;
		}
		
		@Override
		public double partialDistanceToDistance(double d) {
			return FastMath.sqrt(d);
		}
		
		@Override
		public String getName() {
			return "Euclidean";
		}
	},
	
	
	BRAY_CURTIS {
		
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			
			final int n = a.length;
			double sum_1 = 0, sum_2 = 0;
			for(int i = 0; i < n; i++) {
				sum_1 += FastMath.abs(a[i] - b[i]);
				sum_2 += FastMath.abs(a[i] + b[i]);
			}
			
			return sum_1 / sum_2;
		}
		
		@Override
		final public double getP() {
			return DEFAULT_P;
		}
		
		@Override
		public String getName() {
			return "BrayCurtis";
		}
	},
	
	
	CANBERRA {
		
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			
			final int n = a.length;
			double sum=0;
			for(int i = 0; i < n; i++) {
				sum += ( FastMath.abs(a[i] - b[i]) /
					(FastMath.abs(a[i]) + FastMath.abs(b[i])) );
			}
			
			return sum;
		}
		
		@Override
		final public double getP() {
			return DEFAULT_P;
		}
		
		@Override
		public String getName() {
			return "Canberra";
		}
	},
	
	
	
	CHEBYSHEV {
		
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a, b);
			
			final int n = a.length;
			double max = 0;
			for(int i = 0; i < n; i++) {
				double abs = FastMath.abs(a[i] - b[i]);
				if(abs > max)
					max = abs;
			}
			
			return max;
		}
		
		@Override
		final public double getP() {
			return Double.POSITIVE_INFINITY;
		}
		
		@Override
		public String getName() {
			return "Chebyshev";
		}
	},
	
	
	DICE {
		
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			BooleanSimilarity bool = BooleanSimilarity.build(a, b);
			double ctt = bool.one, ctf = bool.two, cft = bool.three;
			
			// If all values in a and b are 0s, the distance will be NaN.
			// Do we want to call that a distance of positive infinity? Or zero?
			return filterNaN((ctf + cft) / (2 * ctt + cft + ctf));
		}
		
		@Override
		final public double getP() {
			return DEFAULT_P;
		}
		
		@Override
		public String getName() {
			return "Dice";
		}
	},
	
	
	KULSINSKI {
		
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			BooleanSimilarity bool = BooleanSimilarity.build(a, b);
			final double ctt = bool.one, ctf = bool.two, cft = bool.three;
			
			return (ctf + cft - ctt + a.length) / (cft + ctf + a.length);
		}
		
		@Override
		final public double getP() {
			return DEFAULT_P;
		}
		
		@Override
		public String getName() {
			return "Kulsinski";
		}
	},
	
	
	ROGERS_TANIMOTO {
		@Override
		public double getPartialDistance(final double[]a, final double[] b) {
			BooleanSimilarity bool = BooleanSimilarity.build(a, b);
			final double ctt = bool.one, ctf = bool.two, cft = bool.three, cff = bool.four;
			final double R = 2 * (cft + ctf);
			return R / (ctt + cff + R);
		}
		
		@Override
		final public double getP() {
			return DEFAULT_P;
		}
		
		@Override
		public String getName() {
			return "RogersTanimoto";
		}
	},
	
	
	RUSSELL_RAO {
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			// This actually takes 3N and can get expensive...
			/*final double ip = VecUtils.innerProduct(
				BooleanSimilarity.asBool(a), 
				BooleanSimilarity.asBool(b));*/
			
			BooleanSimilarity bool = BooleanSimilarity.build(a, b);			
			final double n = (double)a.length;
			return (n - bool.one) / n;
		}
		
		@Override
		final public double getP() {
			return DEFAULT_P;
		}
		
		@Override
		public String getName() {
			return "RussellRao";
		}
	},
	
	
	SOKAL_SNEATH {
		@Override
		final public double getP() {
			return DEFAULT_P;
		}
		
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			BooleanSimilarity bool = BooleanSimilarity.build(a, b);
			final double ctt = bool.one, ctf = bool.two, cft = bool.three;
			final double R = 2 * (cft + ctf);

			// If all values in a and b are 0s, the distance will be NaN.
			// Do we want to call that a distance of positive infinity? Or zero?
			return filterNaN(R / (ctt + R));
		}
		
		@Override
		public String getName() {
			return "SokalSneath";
		}
	},
	
	/*
	YULE {
		@Override
		final public double getP() {
			return DEFAULT_P;
		}
		
		@Override
		public double getPartialDistance(final double[] a, final double[] b) {
			BooleanSimilarity bool = BooleanSimilarity.build(a, b);
			final double ctt = bool.one, ctf = bool.two, cft = bool.three, cff = bool.four;
			final double R = 2 * cft * ctf;
			final double v = R / (ctt * cff + (cft * ctf));

			// If all values in a and b are 0s, the distance will be NaN.
			// Do we want to call that a distance of positive infinity? Or zero?
			return filterNaN(v);
		}
		
		@Override
		public String getName() {
			return "Yule";
		}
	}
	*/
	
	;
	
	
	/**
	 * Should be used on distance methods which may have a 0
	 * in the denominator, creating a distance of {@link Double#NaN}. This will
	 * return a distance of {@link Double#POSITIVE_INFINITY} instead.
	 * @param a
	 * @return infinity if NaN, the distance otherwise.
	 */
	private static double filterNaN(double a) {
		return Double.isNaN(a) ? Double.POSITIVE_INFINITY : a;
	}
	
	@Override
	public double getDistance(double[] a, double[] b) {
		return partialDistanceToDistance(getPartialDistance(a, b));
	}

	@Override
	public double partialDistanceToDistance(double d) {
		return d;
	}

	@Override
	public double distanceToPartialDistance(double d) {
		return d;
	}
}
